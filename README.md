# gguf_offload

**gguf_offload** is a lightweight inference framework that combines GGUF quantization with asynchronous offloading to enable efficient LLM inference. By leveraging lazy loading from GGUF, GPU dequantization, and pipelining, this tool minimizes GPU memory usage—making it possible to run large models with limited GPU resources.

## Performance Considerations

LLM inference with CPU offloading is often bottlenecked by two major factors:

1. **PCIe Bandwidth:** Transferring large amounts of data (such as fully dequantized model weights) between the host and the GPU can saturate the PCIe bus, creating a significant performance bottleneck.
2. **GPU Memory:** Fully loading and storing dequantized model parameters on the GPU may exceed available memory, especially for large language models.

By performing **dequantization directly on the GPU**, our approach mitigates these bottlenecks. Here's how:

- **Reduced Data Transfer:** The model is stored in a quantized format, which takes up much less space. Only minimal quantized representations are transferred over the PCIe bus.
- **On-GPU Processing:** Once on the GPU, dequantization is executed in parallel, converting the quantized data into the required full-precision format. This eliminates the need for transferring large, dequantized datasets between the host and GPU.

## Dependency
https://github.com/chu-tianxiang/llama-cpp-torch

`pip install torch huggingface gguf accelerate`

## Usage

To run the DeepSeek V3/R1 model with minimal GPU memory usage, simply execute:
```bash
python lazy_deepseek.py
```
This script demonstrates:

- Lazy Loading: Only loads model parts as needed.
- GPU Dequantization: Converts quantized weights on the GPU.

## Benchmark
Using batchsize=1, seq_len=512, prefilling

For Qwen-0.5B
- Full GPU inference: 0.030s (2348 MiB)
- gguf_offload: 0.075s (1086 MiB)
- sequential offload using `accelerate`: 0.293s

For DeepSeek-R1 671B 1.58bit quantization
- gguf_offload: ~14s prefill, ~0.5s decode (bounded by pcie bandwidth, should be around 0.3 on a proper pcie 4.0 machine)

Update: With caching frequently used experts, we can achieve 4-5 token/s!

## Potential bugs
Surge GPU memory usage

## TODO
Support end to end generation

Happy Inference!
