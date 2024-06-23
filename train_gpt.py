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

        # attention - materialize the large (T, T) matrix for all keys, queries
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # multiply query keys
        att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float('-inf')) # removing lower half triangle
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.config = config

        # create a wrapper to follow transformers architecture
        self.transformers = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embeddings - layer used to embed tokens (words or subword units) into a continuous vector space of dimension 
            wpe = nn.Embedding(config.block_size, config.n_embd), # word positional embeddings - layer used for representing positions of tokens within a sequence
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformers block
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme - number of parameters to learn is reduced
        # embedding of a token should be closely related to how the token is predicted in the output
        self.transformers.wte.weight = self.lm_head

        self.apply(self._init_weights)

    def _init_weights(self,module):
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

        


