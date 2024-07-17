from dataclasses import dataclass
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

@dataclass
class Config:
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulate_every: int = 4
    learning_rate: float = 2e-4
    validate_every: int = 50
    generate_every: int = 25
    prime_len: int = 100
    seq_len: int = 1024
    segment_length: int = 128
    tokenizer_name: str = 'gpt2'
    dataset_name: str = 'Salesforce/wikitext'
    dataset_version: str = 'wikitext-2-raw-v1'
    num_tokens: int = 50265  # Typical vocab size for BPE
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8
    use_mem_delta_rule: bool = True
    print_every: int = 10
    wandb_project: str = "infini-transformer-pytorch"

def main(config: Config):

    wandb.init(project=config.wandb_project, config=config)

    print(config)

    print(f"Effective batch size: {config.batch_size * config.gradient_accumulate_every}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Load dataset
    dataset = load_dataset(config.dataset_name, config.dataset_version)
    train_data = dataset['train']
    valid_data = dataset['validation']

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=config.seq_len)

    train_data = train_data.map(tokenize_function, batched=True)
    valid_data = valid_data.map(tokenize_function, batched=True)

    # Prepare PyTorch datasets
    class TokenizedDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])

    train_dataset = TokenizedDataset(train_data)
    val_dataset = TokenizedDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # helpers
    def decode_token(token):
        return tokenizer.decode(token)

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
        detach_mems_every_num_segments=2
    ).cuda()

    # optimizer
    optim = Adam(model.parameters(), lr=config.learning_rate)

    # training
    for epoch in trange(config.num_epochs, desc="Epochs"):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc="Batches", mininterval=10.)):
            for __ in range(config.gradient_accumulate_every):
                input_ids = batch['input_ids'].cuda()
                loss = wrapper(
                    input_ids,
                    backward=True,
                    grad_accum_scale=config.gradient_accumulate_every ** -1.
                )

            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

            if i % config.print_every == 0:
                print(f'Step {i}, Training loss: {loss.item()}')
                wandb.log({"training_loss": loss.item(), "step": i + epoch * len(train_loader)})

            if i % config.validate_every == 0:
                with torch.no_grad():
                    model.eval()
                    val_batch = next(iter(val_loader))
                    val_loss = wrapper(val_batch['input_ids'].cuda())
                    print(f'Validation loss: {val_loss.item()}')
                    wandb.log({"validation_loss": val_loss.item(), "step": i + epoch * len(train_loader)})
                    model.train()

            if i % config.generate_every == 0:
                ids = next(iter(val_loader))['input_ids'][:, :config.prime_len].cuda()
                prime = decode_tokens(ids.flatten().tolist())
                print('%s \n\n %s' % (prime, '*' * 100))

                sample = wrapper.generate(
                    prompt=ids,
                    seq_len=config.seq_len
                )

                decoded_string = decode_tokens(sample.flatten().tolist())
                print(decoded_string)
                print("\n")
                wandb.log({"generated_text": decoded_string, "step": i + epoch * len(train_loader)})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}, Training Loss: {avg_epoch_loss}')
        wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

    wandb.finish()

if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)
