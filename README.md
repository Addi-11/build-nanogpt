# Building GPT-2
- Inspired from : [Andrej-Karpathy NanoGPT](https://github.com/karpathy/build-nanogpt)
- Video Lecture: [Let's reproduce GPT-2 (124M) YouTube lecture](https://youtu.be/l8pRSuU81PU)

## Notes
![transformers-architecture](pics/image.png)
#### Optimization Steps taken: 
1. Weight sharing schemes : wte and lm_head have same embeddings, input and output embeddings usually same: synonymns have same probabilites
2. Weight initialization: with std = 0.02 and conditional scaling for deeper networks
3. Controlling precision of matrix multiplication, with float32. Ensure higher accuracy in matrix multiplication operations, at the potential cost of slower performance. : <br>`torch.set_float32_matmul_precision('high')`
4. Typecast float to bfloat16
    - 1 bit for the sign.
    - 8 bits for the exponent.
    - 7 bits for the fraction (mantissa).
    - Range and Precision: The exponent in BFLOAT16 is the same as in FP32, which means it has the same range (from very small to very large numbers). However, the precision is lower due to having fewer bits for the fraction.
    - BFLOAT16 uses half the memory of FP32 
    - Modern GPUs and TPUs often have optimized paths for BFLOAT16 arithmetic.
5. Add torch.compile() : Ahead-Of-Time (AOT) compilation techniques = faster execution times
     - convert model to optimized intermediate representation - fusees multiple small operations into single larger one, reducing overhead of launching kernels on hardware
6. Switch to Flash Attention :
     - Kernel Fusion: By fusing multiple steps of the attention calculation into a single kernel, FlashAttention reduces the overhead of launching multiple separate kernels, leading to faster execution times.
- variances in residual stream grows, so scaling factor 1/sqrt(n), to control activations
- every layer in traansformers has 2 blocks that add to residual networks.
## Increasing Training Speed
- use int8 for inferencing not 
2. TensorCore NVIDIA PAPER: 
- pytorch autocast, turning in bfloat16
- torch.compile() - reduces to-s and fros between memory, although can't find FLASH Attention
- Kernel Fusion
- Avoid Ugly numbers, use power of 2's
- driving probablities to zeros, for extra tokens to you have which you know arent in dataset, drive them to zero.
- Gradient Clipping, to prevent model shock
- Cosine decay learning scheduler
- weight decay parameter - done for parameters in participating in matrix multiplication and embeddings : optimizing in AdamW
- Gradient accumulation, in paralled distribution

