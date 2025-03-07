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

https://github.com/Isotr0py/ggml-libtorch

```bash
pip install torch huggingface gguf accelerate
git clone https://github.com/huggingface/transformers
pip install -e transformers
```

## Usage

To run the DeepSeek V3/R1 model with minimal GPU memory usage, simply execute:
```bash
python lazy_deepseek.py
```
This script demonstrates:

- Lazy Loading: Only loads model parts as needed.
- GPU Dequantization: Converts quantized weights on the GPU.

## Benchmark

Deleted for now due to numerical issues in previous kernel

## Potential bugs
Surge GPU memory usage

## TODO
Support end to end generation

Happy Inference!
