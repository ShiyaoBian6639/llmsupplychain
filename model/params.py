from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # original is 50257, but it is not multiple of powers of 2
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
