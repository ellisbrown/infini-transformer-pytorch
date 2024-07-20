from dataclasses import dataclass
import os
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
from tqdm import tqdm, trange
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
import wandb
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from itertools import chain, islice
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.distributed as dist

from ezcolorlog import root_logger as logger


@dataclass
class Config:
    num_epochs: int = 10
    batch_size: int = 4  # per-device batch size
    gradient_accumulate_every: int = 4
    learning_rate: float = 2e-4
    validate_every: int = 50
    val_steps: int = 100
    generate_every: int = 50
    prime_len: int = 100
    seq_len: int = 1024
    gen_seq_len: int = 1024
    segment_length: int = 128
    tokenizer_name: str = 'gpt2'
    dataset_name: str = 'Salesforce/wikitext'
    dataset_version: str = 'wikitext-2-raw-v1'
    depth: int = 8
    dim: int = 512
    dim_head: int = 64
    heads: int = 8
    use_mem_delta_rule: bool = True
    print_every: int = 10
    wandb_project: str = "infini-transformer-pytorch"
    wandb_run_name: str = None
    seed: int = 42
    save_dir: str = "checkpoints"
    save_every: int = 1 # save every n epochs
    precision: str = "bf16"


class StreamingLMDataset(IterableDataset):
    def __init__(self, dataset, block_size):
        self.dataset = dataset
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            buffer.extend(example['input_ids'])
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size])
                buffer = buffer[self.block_size:]


def main(config: Config):

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=config.precision)
    device = accelerator.device

    config.effective_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulate_every
    
    # Set seed for reproducibility
    set_seed(config.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    config.num_tokens = len(tokenizer)  # Dynamically set num_tokens

    logger.info(f"Current process index: {accelerator.process_index},\t"
                f"Device: {device}")

    if accelerator.is_main_process:
        logger.warning(f"Distributed type: {accelerator.distributed_type}")
        logger.warning(f"Number of processes: {accelerator.num_processes}")
        logger.warning(f"Effective batch size: {config.effective_batch_size}")

        # Initialize wandb
        wandb.init(project=config.wandb_project, config=config, name=config.wandb_run_name)
        logger.info(f"Config: {config}")

    # Load dataset
    dataset_args = dict(
        path=config.dataset_name,
        name=config.dataset_version,
        trust_remote_code=True,
        streaming=True
    )

    train_dataset = load_dataset(split='train', **dataset_args)
    validation = load_dataset(split='validation', **dataset_args)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_attention_mask=False, truncation=False)

    # Tokenize datasets
    tok_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tok_val = validation.map(tokenize_function, batched=True, remove_columns=['text'])


    # Apply group_texts to the datasets
    train_data = tok_train.select_columns('input_ids')
    val_data = tok_val.select_columns('input_ids')

    train_dataset = StreamingLMDataset(train_data, config.seq_len)
    val_dataset = StreamingLMDataset(val_data, config.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    # helpers
    def decode_tokens(tokens):
        return tokenizer.decode(tokens)

    # instantiate GPT-like decoder model
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

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # training
    steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs", disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0
        i = 0
        for i, batch in enumerate(tqdm(train_loader, desc="Batches", mininterval=10., disable=not accelerator.is_main_process)):
            with accelerator.accumulate(wrapper):
                loss = wrapper(
                    batch,
                    backward=True,
                    grad_accum_scale=1 / config.gradient_accumulate_every
                )
                accelerator.clip_grad_norm_(wrapper.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            # accumulate loss
            epoch_loss += loss.item()
            
            # log
            """NOTE:
            had issues when only main proc did gen/val, so moved it to all procs
            but still only log from main proc
            """
            logs = {}

            if i % config.print_every == 0:
                logger.info(f'[i={i}]\tTraining loss: {loss.item()}')
                logs["training_loss"] = loss.item()
                logs["avg_epoch_loss"] = epoch_loss / (i + 1)

            if i % config.validate_every == 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                with torch.no_grad():
                    for val_batch in tqdm(islice(val_loader, config.val_steps), desc=f"[i={i}] Validation", mininterval=10., disable=not accelerator.is_main_process):
                        val_loss += wrapper(val_batch).item()
                        val_steps += 1
                val_loss /= val_steps
                logs["validation_loss"] = val_loss
                model.train()

            if i % config.generate_every == 0:
                model.eval()
                # get a random batch for generation
                gen_batch = next(iter(val_loader))
                prompt = gen_batch[:1, :config.prime_len].to(device)
                generated = wrapper.generate(
                    prompt=prompt,
                    seq_len=config.gen_seq_len
                )
                # log the shapes
                decoded = tokenizer.decode(generated[0])
                logs["generated_text"] = f"PROMPT:\n{decode_tokens(prompt[0])}\n\nGENERATED: {decoded}"
                model.train()
            
            if len(logs) > 0:
                logs["step"] = step
                if accelerator.is_main_process:
                    log_str = "\n\t - ".join([f"{k}: {v}" for k, v in logs.items()])
                    logger.warning(f"[i={i}]\tLogs:\n\t - {log_str}")
                    wandb.log(logs, step=step)

                # logger.info(f'[i={i}]\tWaiting for everyone...')
                accelerator.wait_for_everyone()
                # logger.info(f'[i={i}]\tContinuing...')

            steps += 1

        # Calculate and log average epoch loss
        avg_epoch_loss = epoch_loss / (i + 1)
        if accelerator.is_main_process:
            logger.info(f'Epoch {epoch + 1}/{config.num_epochs}, Average Training Loss: {avg_epoch_loss}')
            wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

            if (epoch + 1) % config.save_every == 0:
                ckpt_path = os.path.join(config.save_dir, wandb.run.name, f"model_epoch_{epoch + 1}.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                accelerator.save(wrapper.state_dict(), ckpt_path)
                logger.error(f"Model saved at {ckpt_path}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # save the model

        ckpt_path = os.path.join(config.save_dir, wandb.run.name, "final.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # unwrap the model
        model = accelerator.unwrap_model(wrapper.model)
        state_dict = model.state_dict()
        accelerator.save(state_dict, ckpt_path)
        
        wandb.finish()


if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)