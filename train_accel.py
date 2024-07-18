from dataclasses import dataclass
import os
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
from tqdm import tqdm, trange
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain
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
    generate_every: int = 25
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
    seed: int = 42
    save_dir: str = "checkpoints"


class LMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {key: torch.tensor(val) for key, val in item.items()}

    def __len__(self):
        return len(self.dataset)


def main(config: Config):

    # Initialize accelerator
    accelerator = Accelerator()
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
        wandb.init(project=config.wandb_project, config=config)
        logger.info(f"Config: {config}")

    # Load dataset
    dataset = load_dataset(config.dataset_name, config.dataset_version)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_attention_mask=False, truncation=False)

    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=['text'],
        desc="Running tokenizer on dataset",
    )

    block_size = config.seq_len

    # Main data processing function
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size we exclude this batch
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply group_texts to the datasets
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # Prepare PyTorch datasets
    train_data = lm_datasets["train"]
    valid_data = lm_datasets["validation"]

    train_dataset = LMDataset(train_data)
    val_dataset = LMDataset(valid_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # shuffle true on val loader to get random samples for generation
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

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
    for epoch in trange(config.num_epochs, desc="Epochs", disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc="Batches", mininterval=10., disable=not accelerator.is_main_process)):
            with accelerator.accumulate(wrapper):
                input_ids = batch['input_ids']
                loss = wrapper(
                    input_ids,
                    backward=True,
                    grad_accum_scale=1 / config.gradient_accumulate_every
                )
                accelerator.clip_grad_norm_(wrapper.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            # accumulate loss
            epoch_loss += loss.item()

            if accelerator.is_main_process:
                if i % config.print_every == 0:
                    logger.info(f'Step {i}, Training loss: {loss.item()}')
                    wandb.log({"training_loss": loss.item(), "step": i + epoch * len(train_loader)})

                if i % config.validate_every == 0:
                    model.eval()
                    val_loss = 0
                    for val_batch in val_loader:
                        with torch.no_grad():
                            val_input_ids = val_batch['input_ids']
                            val_loss += wrapper(val_input_ids).item()
                    val_loss /= len(val_loader)
                    logger.info(f'Validation loss: {val_loss}')
                    wandb.log({"validation_loss": val_loss, "step": i + epoch * len(train_loader)})
                    model.train()

                if i % config.generate_every == 0:
                    model.eval()
                    # get a random batch for generation
                    gen_batch = next(iter(val_loader))
                    prompt = gen_batch['input_ids'][:1, :config.prime_len].to(device)
                    generated = wrapper.generate(
                        prompt=prompt,
                        seq_len=config.gen_seq_len
                    )
                    decoded = tokenizer.decode(generated[0])
                    logger.info(f"PROMPT:\n{decode_tokens(prompt[0])}\n\nGENERATED: {decoded}\n")
                    wandb.log({"generated_text": decoded, "step": i + epoch * len(train_loader)})
                    model.train()


        # Calculate and log average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        if accelerator.is_main_process:
            logger.info(f'Epoch {epoch + 1}/{config.num_epochs}, Average Training Loss: {avg_epoch_loss}')
            wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save the model
        ckpt_path = os.path.join(config.save_dir, wandb.run.name, "model.pth")
        accelerator.save(wrapper.state_dict(), ckpt_path)
        wandb.finish()


if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)