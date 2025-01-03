import tiktoken
import torch
from torch.nn import functional as F
import time
from transformers import GPT2LMHeadModel
from model.gpt import GPT
from model.params import GPTConfig
from dataloader.mnbvc import DataLoaderZh
from utils.lrscheduler import get_lr

# model_hf = GPT2LMHeadModel.from_pretrained("GPT2")
# sd_hf = model_hf.state_dict()
# for k, v in sd_hf.items():
#     print(k, v.shape)
# training parameters
max_lr = 6e-4
warmup_steps = 10
min_lr = max_lr * 0.1
max_steps = 50
# define device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device: {device}")

# gradient accumulation (deal with low memory gpu)
total_batch_size = 2 ** 19
B = 16
T = 1024
assert total_batch_size % B * T == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size:  {total_batch_size}")
print(f" => calculated gradient accumulation stepts: {grad_accum_steps}")


# load training data
train_loader = DataLoaderZh(B=B, T=T, names=["law_judgement", "wikipedia", "news_peoples_daily"])

torch.set_float32_matmul_precision("high")
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

# optimize
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f} ms "
          f"| tok/sec: {tokens_per_sec:.2f}")

################################
# num_return_sequences = 5
# max_length = 30
#
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("What is house price trend in Beijing?")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to("cuda")
#
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, loss = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 80, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         x_col = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, x_col), dim=1)
#
# for i in range(num_return_sequences):
#     tokens = x[i].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
