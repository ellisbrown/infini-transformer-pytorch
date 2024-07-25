from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple, Union
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
from tqdm import tqdm
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
import wandb
from wandb.sdk.wandb_run import Run
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from itertools import islice
from accelerate import Accelerator
from accelerate.utils import set_seed
from ezcolorlog import root_logger as logger
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import LambdaLR
import math

import logging

# rm the "Running this sequence through the model will result in indexing errors" warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    total_steps: int = 100_000
    batch_size: int = 4  # per-device batch size
    gradient_accumulate_every: int = 4  # per device
    learning_rate: float = 1e-2
    warmup_steps: int = 1000
    val_steps: int = 100
    validate_every: int = 100
    generate_every: int = 100
    prime_len: int = 1024
    seq_len: int = 1024
    gen_seq_len: int = 1024
    segment_length: int = 128
    tokenizer_name: str = 'gpt2'
    dataset_name: str = 'Salesforce/wikitext'
    dataset_version: str = 'wikitext-2-raw-v1'
    dataset_proportions: List[float] = field(default_factory=lambda: [1])
    depth: int = 8
    dim: int = 512
    dim_head: int = 64
    heads: int = 8
    use_mem_delta_rule: bool = True
    print_every: int = 10
    wandb_entity: str = "default"
    wandb_project: str = "infini-transformer-pytorch"
    wandb_run_name: str = None
    seed: int = 42
    save_dir: str = "checkpoints"
    save_every: int = 500  # save every n steps
    precision: str = "bf16"


class StreamingLMDataset(IterableDataset):
    def __init__(self, dataset, block_size):
        self.dataset = dataset
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            input_ids = example['input_ids']
            buffer.extend(input_ids)
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size])
                buffer = buffer[self.block_size:]



def get_train_val_data(dataset_name, dataset_version, tokenize_fn):
    dataset_args = dict(
        path=dataset_name,
        name=dataset_version,
        trust_remote_code=True,
        streaming=True
    )

    train_dataset = load_dataset(split='train', **dataset_args)
    validation = load_dataset(split='validation', **dataset_args)

    # remove potential bad examples
    def filter_fn(x):
        return x is not None and isinstance(x, dict) and 'text' in x and x['text'] is not None and isinstance(x['text'], str) and len(x['text']) > 0

    train_dataset = train_dataset.filter(filter_fn)
    validation = validation.filter(filter_fn)

    tok_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    tok_val = validation.map(tokenize_fn, batched=True, remove_columns=['text'])

    train_dataset = tok_train.select_columns('input_ids')
    val_dataset = tok_val.select_columns('input_ids')
    
    return train_dataset, val_dataset

def get_datasets(dataset_name, dataset_version, tokenize_fn, seq_len, proportions=[1]):
    # handle multiple datasets
    if "-+-" in dataset_name:
        logger.info(f"Loading multiple datasets: {dataset_name} ({dataset_version})")
        if not "-+-" in dataset_version:
            raise ValueError(f"If using multiple datasets, please provide multiple dataset versions. Got: {dataset_version}")
        dataset_names = dataset_name.split("-+-")
        dataset_versions = dataset_version.split("-+-")
        train_datasets = []
        val_datasets = []
        for name, version in zip(dataset_names, dataset_versions):
            train_dataset, val_dataset = get_train_val_data(name, version, tokenize_fn)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        logger.info(f"Loaded {len(train_datasets)} datasets: {dataset_names}")

        if proportions is None or proportions == [1]:
            logger.warning(f"No dataset proportions provided. Defaulting to uniform distribution.")
            train_dataset = interleave_datasets(train_datasets)
            val_dataset = interleave_datasets(val_datasets)

        elif len(proportions) != len(train_datasets):
            raise ValueError(f"Number of dataset proportions ({len(proportions)}) must match number of datasets ({len(train_datasets)})")

        # check if proportions sum to 1 and are non-negative
        elif not math.isclose(sum(proportions), 1.0, rel_tol=1e-5) or any(p < 0 for p in proportions):
            raise ValueError(f"Dataset proportions must sum to 1 and be non-negative. Got: {proportions}")
        
        else:
            logger.info(f"Interleaving datasets {dataset_names} with proportions: {proportions}")
            train_dataset = interleave_datasets(train_datasets, probabilities=proportions)
            val_dataset = interleave_datasets(val_datasets, probabilities=proportions)

    else:
        logger.info(f"Loading dataset: {dataset_name} ({dataset_version})")
        train_dataset, val_dataset = get_train_val_data(dataset_name, dataset_version, tokenize_fn)

    lm_train_dataset = StreamingLMDataset(train_dataset, seq_len)
    lm_val_dataset = StreamingLMDataset(val_dataset, seq_len)
    return lm_train_dataset, lm_val_dataset


def cosine_with_warmup_lr_scheduler(optimizer, warmup_steps, train_steps, cycles=0.5, last_epoch=-1):
    """
    Cosine learning rate scheduler with warm-up.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): The number of warm-up steps.
        train_steps (int): The total number of training steps.
        cycles (float, optional): The number of cycles for the cosine annealing schedule. Default is 0.5.
        last_epoch (int, optional): The index of the last epoch. Default is -1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, train_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(cycles * math.pi * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def get_latest_checkpoint(config: Config, run: Run) -> Tuple[Union[Path, None], bool]:
    """
    Get the latest checkpoint file for a given run.

    Args:
        config (Config): The configuration object.
        run (Run): The wandb run object.

    Returns:
        Tuple[Union[Path, None], bool]: A tuple containing the latest checkpoint file path and a boolean indicating
        whether the checkpoint is local or fetched from wandb. If no checkpoint is found, returns (None, False).
    """
    
    checkpoint_dir = Path(config.save_dir) / run.name
    local_checkpoints = sorted(checkpoint_dir.glob("model_step_*.pth"), key=lambda x: int(x.stem.split('_')[-1]))
    
    if local_checkpoints:
        return local_checkpoints[-1], True  # Local checkpoint exists
    
    # If no local checkpoints, try to fetch from wandb
    wandb_files = run.files()
    checkpoint_files = [file for file in wandb_files if file.name.startswith("model_step_") and file.name.endswith(".pth")]
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
        return latest_checkpoint, False  # Wandb checkpoint exists
    
    return None, False  # No checkpoint found


def main(config: Config):
    accelerator = Accelerator(mixed_precision=config.precision)
    device = accelerator.device

    config.effective_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulate_every
    
    set_seed(config.seed)

    if config.tokenizer_name.startswith("tiktoken-"):
        import tiktoken
        tiktoken_name = config.tokenizer_name.replace("tiktoken-", "")
        logger.info(f"Using tiktoken tokenizer: {tiktoken_name}")
        tokenizer = tiktoken.get_encoding(tiktoken_name)
        config.num_tokens = tokenizer.n_vocab
        
        def tokenize_function(examples):
            try:
                input_ids = tokenizer.encode_ordinary_batch(examples['text'])
                return dict(input_ids=input_ids)
            except Exception as e:
                logger.error(f"Error tokenizing: {len(examples['text'])} examples")
                raise e
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        config.num_tokens = len(tokenizer)

        def tokenize_function(examples):
            try:
                return tokenizer(examples['text'], return_attention_mask=False, truncation=False)
            except Exception as e:
                logger.error(f"Error tokenizing: {len(examples['text'])} examples")
                raise e

    def decode_tokens(tokens):
        return tokenizer.decode(tokens)


    logger.info(f"Current process index: {accelerator.process_index}, Device: {device}")


    if accelerator.is_main_process:
        # Check if a run with the same name exists
        api = wandb.Api()
        entity = wandb.Settings().get('default', 'entity')
        runs = api.runs(f"{config.wandb_entity}/{config.wandb_project}")
        existing_run = next((run for run in runs if run.name == config.wandb_run_name), None)

        if existing_run:
            logger.info(f"Resuming existing run: {config.wandb_run_name}")
            run = wandb.init(project=config.wandb_project, id=existing_run.id, resume="must")
        else:
            logger.info(f"Starting new run: {config.wandb_run_name}")
            run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

        logger.info(f"Config: {config}")

    train_dataset, val_dataset = get_datasets(config.dataset_name, config.dataset_version, tokenize_function, config.seq_len, config.dataset_proportions)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    model = InfiniTransformer(
        num_tokens=config.num_tokens,
        dim=config.dim,
        depth=config.depth,
        dim_head=config.dim_head,
        heads=config.heads,
        use_mem_delta_rule=config.use_mem_delta_rule
    )

    wrapper = InfiniTransformerWrapper(
        model,
        segment_length=config.segment_length,
        detach_mems_every_num_segments=2,
        accelerator=accelerator
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer = Adafactor(
    #     model.parameters(),
    #     scale_parameter=False,
    #     relative_step=False,
    #     warmup_init=False,
    # )

    lr_scheduler = cosine_with_warmup_lr_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        train_steps=config.total_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=config.print_every)

    # Always try to restore the latest checkpoint
    start_step = 0
    if accelerator.is_main_process:
        latest_checkpoint, is_local = get_latest_checkpoint(config, run)
        
        if latest_checkpoint:
            if is_local:
                logger.info(f"Found local checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint)
            else:
                logger.info(f"Found wandb checkpoint: {latest_checkpoint.name}")
                checkpoint_file = latest_checkpoint.download(replace=True)
                checkpoint = torch.load(checkpoint_file.name)
            
            wrapper.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_step = checkpoint['step']
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.info("No checkpoint found, starting from beginning")

    # Broadcast the start_step to all processes
    start_step = torch.distributed.broadcast(start_step, src=0)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    total_loss = 0
    pbar = tqdm(range(config.total_steps), desc="Training", disable=not accelerator.is_main_process)
    for step in pbar:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(wrapper):
            loss = wrapper(
                batch,
                backward=True,
                grad_accum_scale=1 / config.gradient_accumulate_every
            )
            accelerator.clip_grad_norm_(wrapper.parameters(), 0.5)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        
        logs = {}

        if step % config.print_every == 0:
            avg_loss = total_loss / config.print_every
            logger.info(f'[Step={step}]\tAverage Training loss: {avg_loss}')
            logs["training_loss"] = avg_loss
            total_loss = 0

        if step % config.validate_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(config.val_steps):
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    val_loss += wrapper(val_batch).item()
            val_loss /= config.val_steps
            logs["validation_loss"] = val_loss
            model.train()

        if step % config.generate_every == 0:
            model.eval()
            try:
                gen_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                gen_batch = next(val_iter)
            prompt = gen_batch[:1, :config.prime_len].to(device)
            generated = wrapper.generate(
                prompt=prompt,
                seq_len=config.gen_seq_len
            )
            decoded = decode_tokens(generated[0])
            logs["generated_text"] = f"PROMPT:\n{decode_tokens(prompt[0])}\n\nGENERATED: {decoded}"
            model.train()
        
        if logs and accelerator.is_main_process:
            logs["step"] = step
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            log_str = "\n\t - ".join([f"{k}: {v}" for k, v in logs.items()])
            logger.warning(f"[Step={step}]\tLogs:\n\t - {log_str}")
            wandb.log(logs, step=step)

        if step % config.save_every == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(config.save_dir, wandb.run.name, f"model_step_{step}.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            checkpoint = {
                'step': step,
                'model_state_dict': accelerator.unwrap_model(wrapper.model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }
            accelerator.save(checkpoint, ckpt_path)
            wandb.save(ckpt_path, base_path=config.save_dir)
            logger.error(f"Model saved at {ckpt_path}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save final checkpoint
        ckpt_path = os.path.join(config.save_dir, wandb.run.name, f"model_step_{config.total_steps}.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        checkpoint = {
            'step': config.total_steps,
            'model_state_dict': accelerator.unwrap_model(wrapper.model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        accelerator.save(checkpoint, ckpt_path)
        wandb.save(ckpt_path, base_path=config.save_dir)
        wandb.finish()

if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)