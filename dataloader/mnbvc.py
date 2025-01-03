import tiktoken
import torch
import os

from datasets import load_dataset


class DataLoaderZh:
    def __init__(self, B, T, names):
        self.B = B
        self.T = T
        self.names = names
        self.name_ptr = 0
        self.name = self.get_name()
        self.enc = tiktoken.get_encoding("gpt2")

        self.dataloader = self.load_data()
        tokens = self.get_tokens()
        self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def load_data(self):
        return iter(load_dataset("liwu/MNBVC", name=self.name, split='train', streaming=True))

    def get_name(self):
        return self.names[self.name_ptr]

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        # bound handling
        if self.current_position + B * T + 1 > len(self.tokens):
            # calculate remaining tokens
            remaining_tokens = self.tokens[self.current_position:]
            # update tokens

            new_tokens = torch.tensor(self.get_tokens())
            self.tokens = torch.cat((remaining_tokens, new_tokens), dim=0)
            self.current_position = 0

        return x, y

    def get_text(self, data):
        if self.name in ["law_judgement"]:
            text = data["text"]
        elif self.name in ["news_peoples_daily", "wikipedia"]:
            text = parse_data(data)
        else:
            text = ""
        return text

    def get_tokens(self):
        res = []
        for _ in range(10000):
            try:
                data = next(self.dataloader)
                text = self.get_text(data)
                res.append(text)
            except StopIteration:
                if self.name_ptr + 1 <= len(self.names):
                    self.name_ptr += 1
                    self.name = self.get_name()
                break
        tokens = self.enc.encode(" ".join(res))
        return tokens


def parse_data(data: dict) -> str:
    res = []
    for paragraph in data["段落"]:
        res.append(paragraph["内容"])
    return " ".join(res)


ld = DataLoaderZh(4, 32, ["wikipedia"])
for i in range(1000000):
    x, y = ld.next_batch()