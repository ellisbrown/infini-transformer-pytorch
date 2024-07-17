from dataclasses import dataclass
from infini_transformer_pytorch import (
    InfiniTransformer,
    InfiniTransformerWrapper
)
import tqdm
import gzip
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

@dataclass
class Config:
    num_batches: int = int(1e5)
    batch_size: int = 4
    gradient_accumulate_every: int = 4
    learning_rate: float = 2e-4
    validate_every: int = 100
    generate_every: int = 250
    prime_len: int = 100
    seq_len: int = 1024
    segment_length: int = 128
    data_path: str = './data/enwik8.gz'
    num_tokens: int = 256
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8
    use_mem_delta_rule: bool = True


def main(config: Config):

    print(config)

    print(f"Effective batch size: {config.batch_size * config.gradient_accumulate_every}")

    # helpers
    def cycle(loader):
        while True:
            for data in loader:
                yield data

    def decode_token(token):
        return str(chr(max(32, token)))

    def decode_tokens(tokens):
        return ''.join(list(map(decode_token, tokens)))

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

    # prepare enwik8 data
    with gzip.open(config.data_path) as file:
        x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(x, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len].long()
            return full_seq.cuda()

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, config.seq_len)
    val_dataset = TextSamplerDataset(data_val, config.seq_len)
    train_loader = cycle(DataLoader(train_dataset, batch_size=config.batch_size))
    val_loader = cycle(DataLoader(val_dataset, batch_size=1))

    # optimizer
    optim = Adam(model.parameters(), lr=config.learning_rate)

    # training
    for i in tqdm.tqdm(range(config.num_batches), mininterval=10.):

        for __ in range(config.gradient_accumulate_every):
            loss = wrapper(
                next(train_loader),
                backward=True,
                grad_accum_scale=config.gradient_accumulate_every ** -1.
            )

        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % config.validate_every == 0:
            with torch.no_grad():
                wrapper.eval()
                loss = wrapper(next(val_loader))
                print(f'validation loss: {loss.item()}')

        if i % config.generate_every == 0:
            ids = next(val_loader)[:, :config.prime_len]
            prime = decode_tokens(ids.flatten())
            print('%s \n\n %s', (prime, '*' * 100))

            sample = wrapper.generate(
                prompt=ids,
                seq_len=config.seq_len
            )

            decoded_string = decode_tokens(sample.flatten())
            print(decoded_string)
            print("\n")

if __name__ == "__main__":
    from jsonargparse import CLI
    args = CLI(Config, as_positional=False)
    main(args)
