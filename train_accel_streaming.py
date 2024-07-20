from dataclasses import dataclass
import os
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
import wandb
from datasets import load_dataset
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


@dataclass
class Config:
    total_steps: int = 100000
    batch_size: int = 4  # per-device batch size
    gradient_accumulate_every: int = 4
    learning_rate: float = 1e-2
    warmup_steps: int = 1000
    max_lr: float = 0.01
    min_lr: float = 1e-5
    validate_every: int = 100
    val_steps: int = 100
    generate_every: int = 100
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
    save_every: int = 5000  # save every n steps
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


def get_lr_scheduler(optimizer, warmup_steps, total_steps, max_lr, min_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress))) * (max_lr - min_lr) + min_lr

    return LambdaLR(optimizer, lr_lambda)


def main(config: Config):
    accelerator = Accelerator(mixed_precision=config.precision)
    device = accelerator.device

    config.effective_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulate_every
    
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    config.num_tokens = len(tokenizer)

    logger.info(f"Current process index: {accelerator.process_index}, Device: {device}")

    if accelerator.is_main_process:
        logger.warning(f"Distributed type: {accelerator.distributed_type}")
        logger.warning(f"Number of processes: {accelerator.num_processes}")
        logger.warning(f"Effective batch size: {config.effective_batch_size}")
        wandb.init(project=config.wandb_project, config=config, name=config.wandb_run_name)
        logger.info(f"Config: {config}")

    dataset_args = dict(
        path=config.dataset_name,
        name=config.dataset_version,
        trust_remote_code=True,
        streaming=True
    )

    train_dataset = load_dataset(split='train', **dataset_args)
    validation = load_dataset(split='validation', **dataset_args)

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_attention_mask=False, truncation=False)

    tok_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tok_val = validation.map(tokenize_function, batched=True, remove_columns=['text'])

    train_data = tok_train.select_columns('input_ids')
    val_data = tok_val.select_columns('input_ids')

    train_dataset = StreamingLMDataset(train_data, config.seq_len)
    val_dataset = StreamingLMDataset(val_data, config.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    def decode_tokens(tokens):
        return tokenizer.decode(tokens)

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

    lr_scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        max_lr=config.max_lr,
        min_lr=config.min_lr
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    total_loss = 0
    for step in tqdm(range(config.total_steps), desc="Training", disable=not accelerator.is_main_process):
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
            decoded = tokenizer.decode(generated[0])
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
            accelerator.save(wrapper.state_dict(), ckpt_path)
            logger.error(f"Model saved at {ckpt_path}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        ckpt_path = os.path.join(config.save_dir, wandb.run.name, "final.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model = accelerator.unwrap_model(wrapper.model)
        state_dict = model.state_dict()
        accelerator.save(state_dict, ckpt_path)
        wandb.finish()

if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)