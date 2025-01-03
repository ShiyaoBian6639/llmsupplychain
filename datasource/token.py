import tiktoken
import os

enc = tiktoken.get_encoding("gpt2")
name = "law_judgement"
files = os.listdir(f"./data/{name}")
for file in files[:2]:
    with open(f"./data/{name}/{file}", 'r') as f:
        text = f.read()
    tokens = enc.encode(text)
# tokens = enc.encode("What is house price trend in Beijing?")
