# Building GPT-2
1. Reproduced and optimized a GPT-2 124M model using transformers-based architecture, implementing several advanced techniques to enhance performance and efficiency, following Andrej Karpathy's lecture series.

2. Following the original paper, added technical improvements such as weight sharing, Adam optimizer, and gradient clipping.

3. Enhancements: Employed torch.compile() for AOT compilation, reducing kernel launch overhead and enhancing execution speed. Adjusted vocab size from 50257 to 50304 using power-of-two padding for computational efficiency. Implemented precision control with float32 and bfloat16, FlashAttention, AdamW optimizer, cosine decay learning scheduler, weight decay, and gradient accumulation.

4. Achieved a HellaSwag score of 0.3337 and validation loss of 2.9478 using 4 V100 GPUs, compared to the original GPT-2 124M validation loss of 3.12, after training for 2 full days.
 
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

| **MapReduce**                     | **Transformer Attention Equivalent**                                                                                        |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `map(x) → (k, v)`                           | Each input vector `xᵢ` produces a **key** `kᵢ = xᵢW_K` and a **value** `vᵢ = xᵢW_V`.                                        |
| `query` not in MapReduce                    | Each position `i` in the input also computes a **query** `qᵢ = xᵢW_Q`.                                                      |
| `shuffle and group by key`                  | Soft matching: For each `qᵢ`, compute dot-product similarity with all keys `kⱼ`, yielding weights `αᵢⱼ = softmax(qᵢ ⋅ kⱼ)`. |
| `reduce` by aggregating values for each key | Attention output `oᵢ = Σⱼ αᵢⱼ ⋅ vⱼ` is a weighted combination of value vectors.                                             |

#### Training Initialization

- Weight intialization with std = 0.02 and conditional scaling for deeper networks. `torch.nn.init.normal_(module.weight, mean=0, std=0.02)`
- Weight intialization using He, Xavier
   - He used in ReLU, Linear, GeLU activations
   - Xiavier used in tanh and sigmoid activations 
- Scaling factor in variances:
   - in the residual stream can grow significantly as the depth of the model increases.
   - scaling factor, to control activations is 1/sqrt(n), n = number of layers, like He Initialization
   - ```
     if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # conditional scaling for deeper models, to control variance
                std *= (2*self.config.n_layer) ** -0.5
     ```


#### Optimization Steps taken: 

1. **Weight sharing schemes :** 
    - wte and lm_head have same embeddings
    - input and output embeddings usually same: synonymns have same probabilites

3. **Precision of matrix multiplication :**
    - control precision with tensorfloat32 (TF32), not default FP32. 
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
    - ```
      # bfloat16 has same exponent range as fp32. 
      # Povides enough precision to maintain model accuracy while reducing computational and memory overhead.
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
      ```

5. **Torch.compile() :**
    - Ahead-Of-Time (AOT) compilation techniques = faster execution times
    - does not use the standard python interpreter - which does not know what comes next.
    - convert model to optimized intermediate representation - fusees multiple small operations into single larger one, reducing overhead of launching kernels on hardware.
    - reduces to-s and fros between memory
    - cant find flash attention though

6. **Switch to Flash Attention :**
    - Kernel Fusion: By fusing multiple steps of the attention calculation into a single kernel, FlashAttention reduces the overhead of launching multiple separate kernels, leading to faster execution times.
    - Intermediate attention state in not materialized.
    - `y = F.scaled_dot_product_attention(q, k, v, is_casual=True)`

7. **Nice Numbers :**
    - vocab size 50257 -> 50304 nice
    - avoid ugly numbers and use power of two's
    - pad vocab_size to nice, then drive probablities to zeros, for extra tokens to you have which you know arent in dataset. These extra tokens though never appear in the dataset, so their probabilities are driven to 0 by softmax.

8. **AdamW params :**
    - `AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)` 
    - beta1: running average of the gradient. Represents the momentum term.
    - beta2: running average of the squared gradient. Represents the adaptive learning rate term. More responsive to recent changes = fast convergence.
    - eps: small constant added to the denominator of the update step to improve numerical stability.

9. **Grad Clipping :**
    - `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
    - prevent model shock, exploding gradient, overshooting gradients

10. **Cosine Decay Learning Scheduler :**
    - reduce the learning rate during training using a cosine function.
    - ```
      # 1) linear warmup for warmup_iters steps
      if it < warmup_steps:
          return max_lr * (it+1) / warmup_steps
      # 2) if it > lr_decay_iters, return min learning rate
      if it > max_steps:
          return min_lr
      # 3) in between, use cosine decay down to min learning rate
      decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
      ```

11. **Weight Decay Paramerter :**
    - done for parameters in participating in matrix multiplication and embeddings
    - prevent overfitting by adding a penalty to the loss function for large weights
    - use fused adam optimizer (hardware efficient)
    - ```
      optimizer = raw_model.configure_params(weight_decay=0.1, learning_rate=6e-4, device=device) # weight decay optimizer
      ```

12. **Gradient Accumulation :**
    - increase batch_size without increasing memory footprint (for 1M params) :  simulate training with a larger effective batch size than what can physically fit in memory (usually GPU VRAM).
    - Instead of updating the model's weights after each mini-batch, gradients are accumulated over multiple mini-batches (called accumulation steps).
    - Instead of updating the weights, add (accumulate) the gradients to a running sum.
    -  Example: 4 forward+backward passes with batch_size=256, accumulating gradients each time, and call optimizer.step() once after all 4. This is mathematically equivalent to doing a single step with batch size 1024.

13. **Distributed Training :**
    - Use DDP (distributed Data Parallel)
    - ```
      ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
      
      if ddp:
          # use of DDP atm demands CUDA, we set the device appropriately according to rank
          assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
          init_process_group(backend='nccl')
          ddp_rank = int(os.environ['RANK'])
          ddp_local_rank = int(os.environ['LOCAL_RANK'])
          ddp_world_size = int(os.environ['WORLD_SIZE'])
          device = f'cuda:{ddp_local_rank}'
          torch.cuda.set_device(device)
          master_process = ddp_rank == 0 
          
      if ddp:
         model = DDP(model, device_ids=[ddp_local_rank])
         raw_model = model.module if ddp else model

      if ddp:
         destroy_process_group()
      ```
14. **Use int8 for inferencing :**
    - we dont need high precision or floating points, so int will work too


### Paper Links:
- [Transformers Architecture: Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [GPT2 Model](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [nvidia-ampere-architecture-whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

