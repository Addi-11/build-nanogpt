from datasets import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not bias, more of a mask but following OpenAI/HF naming
        # mask to remove the lower half triangle of key value matrix multiplication
        # in a attention you focus on values before you, not ahead
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length (context length), embedding dimension (n_embd)
        # calculate query, keys, values for all heads in a batch, and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # In GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) # concatenated query, key, values from x
        q, k, v = qkv.split(self.n_embd, dim=2) # split into each dimension C
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # ATTENTION
        # attention - materialize the large (T, T) matrix for all keys, queries
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # multiply query keys
        # att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float('-inf')) # removing lower half triangle
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # FLASH ATTENTION IMPLEMENTED
        y = F.scaled_dot_product_attention(q, k, v, is_casual=True)
        y = y.transpose(1,2).contigous().view(B,T,C) # re-assemble all head outputs side by side

        y = self.c_proj(y) # output projection
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # expand dimensionality
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.config = config

        # create a wrapper to follow transformers architecture
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embeddings - layer used to embed tokens (words or subword units) into a continuous vector space of dimension 
            wpe = nn.Embedding(config.block_size, config.n_embd), # word positional embeddings - layer used for representing positions of tokens within a sequence
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformers block
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme - number of parameters to learn is reduced
        # embedding of a token should be closely related to how the token is predicted in the output
        self.transformer.wte.weight = self.lm_head

        self.apply(self._init_weights) # to initialize random weights

    def _init_weights(self, module):
        # intializing weights with mean 0, standard deviation 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # conditional scaling for deeper models, to control variance
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is the shape of (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only of {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)

        # forward final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from  huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embd from model type
        config_args = {
            'gpt2' :        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large' :  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl' :     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() # state dictionary
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discarding mask/buffer, 'cause not a param

        # init huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy weights from huggingface, all params are aligned and match names and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # openai checkpoints use conv1D, but our current implementation is linear, which means we transpose these weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf == sd_keys), f"mismatched keys: {len(sd_keys_hf) != len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for convd1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                     sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy for other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy(sd_hf[k])

        return model

# -----------------------------------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        # for x = token[0...3] y = token[4]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        self.current_position += B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0

        return x, y
    
# -----------------------------------------------------------------------------------------------------
# training loop
# auto detect device
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high') # ensure higher accuracy in matrix multiplication, slow performance

# model = GPT.from_pretrained('gpt2')
# print("didn't crash yay!")

# get logits
model = GPT(GPTConfig())
model.to(device) # ensure data and model on same device
model = torch.compile(model)

# learning rate decay - cosine
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    # bfloat16 has same exponent range as fp32. 
    # Povides enough precision to maintain model accuracy while reducing computational and memory overhead.
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()

    # clipp gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # wait for gpu to finish the work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------
# evaluation loop
model.eval()
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate , rn x = (B, T) (5, 8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size() < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)

        probs = F.softmax(logits, dim=-1)
        # do top_k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

