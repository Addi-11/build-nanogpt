# Building GPT-2
- Video Lecture: [Let's reproduce GPT-2 (124M) YouTube lecture](https://youtu.be/l8pRSuU81PU)
- Lecture Repo : [Andrej-Karpathy NanoGPT](https://github.com/karpathy/build-nanogpt)

![transformers-architecture](pics/image.png)

### Transformers Block:
**MHSA**: Captures contextual information from the entire sequence. <br>
**FFN**: Applies non-linear transformations to enrich the feature representations.

```
Input ---> LayerNorm ---> MHSA ---> Add (Residual Connection) ---> LayerNorm ---> FFN ---> Add (Residual Connection) ---> Output
```
adding the outputs of these sub-blocks to the residual stream, the transformer maintains the original input information
```
ln_1 = nn.LayerNorm(config.n_embd)
attn = CasualSelfAttention(config)
ln_2 = nn.LayerNorm(config.n_embd)
mlp = MLP(config)
```

- variances in the residual stream can grow significantly as the depth of the model increases.
- scaling factor, to control activations is 1/sqrt(n), n = number of layers.
- form of normalized initialization like Xavier or He.
- Weight intialization with std = 0.02 and conditional scaling for deeper networks


#### Optimization Steps taken: 

1. **Weight sharing schemes :** 
    - wte and lm_head have same embeddings
    - input and output embeddings usually same: synonymns have same probabilites

3. **Precision of matrix multiplication :**
    - control precision with float32. 
    - Ensure higher accuracy in matrix multiplication operations
    - potential cost of slower performance. 
    - `torch.set_float32_matmul_precision('high')`

4. **Typecast to bfloat16 :**
![alt text](pics/image2.png)
    - 1 bit for the sign.
    - 8 bits for the exponent.
    - 7 bits for the fraction (mantissa).
    - Range and Precision: The exponent in BFLOAT16 is the same as in FP32, which means it has the same range (from very small to very large numbers). However, the precision is lower due to having fewer bits for the fraction.
    - BFLOAT16 uses half the memory of FP32 
    - Modern GPUs and TPUs often have optimized paths for BFLOAT16 arithmetic.

5. **Torch.compile() :**
    - Ahead-Of-Time (AOT) compilation techniques = faster execution times
    - convert model to optimized intermediate representation - fusees multiple small operations into single larger one, reducing overhead of launching kernels on hardware.
    - reduces to-s and fros between memory
    - cant find flash attention though

6. **Switch to Flash Attention :**
    - Kernel Fusion: By fusing multiple steps of the attention calculation into a single kernel, FlashAttention reduces the overhead of launching multiple separate kernels, leading to faster execution times.

7. **Nice Numbers :**
    - vocab size 50257 -> 50304 nice
    - avoid ugly numbers and use power of two's
    - pad vocab_size to nice, then drive probablities to zeros, for extra tokens to you have which you know arent in dataset.

8. **AdamW params :**
    - `AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)` 
    - beta1: running average of the gradient. Represents the momentum term.
    - beta2: running average of the squared gradient. Represents the adaptive learning rate term. More responsive to recent changes = fast convergence.
    - eps: small constant added to the denominator of the update step to improve numerical stability.

9. **Grad Clipping :**
    - prevent model shock

10. **Cosine Decay Learning Scheduler :**

11. **Weight Decay Paramerter :**
    - done for parameters in participating in matrix multiplication and embeddings

12. **Gradient Accumulation in parallel distribution :**

9. **Use int8 for inferencing :**


### Paper Links:
- [Transformers Architecture: Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [GPT2 Model](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [nvidia-ampere-architecture-whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

