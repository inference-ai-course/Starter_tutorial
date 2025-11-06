# Chapter 4: Hugging Face Platform and Local Inference

This chapter covers cloud-based inference with Hugging Face and local deployment using Ollama and vLLM.

**Total Duration:** 10 hours  
**Prerequisites:** Python 3.10+, CUDA 12.4+, PyTorch 2.6.0+

---

## Hugging Face Platform (4 hours)

### 📚 Documentation

- **[01 Hugging Face Overview](01_huggingface_overview.md)** - Introduction and learning path
- **[02 Core Concepts](02_core_concepts.md)** - Inference providers, authentication, APIs
- **[03 Authentication & Security](03_authentication_security.md)** - Token management and best practices
- **[04 Provider Selection](04_provider_selection.md)** - Choosing providers and implementing failover

### 💻 Jupyter Notebooks (Coding Practice)

- **[Image Generation Practice](01_image_generation_practice.ipynb)** - Text-to-image with provider comparison
- **[Chat Inference Practice](02_chat_inference_practice.ipynb)** - Conversational AI and benchmarking

### Learning Objectives

- ✅ Understand Hugging Face Inference Providers and selection strategies
- ✅ Implement secure authentication and credential management
- ✅ Use OpenAI-compatible interfaces with Hugging Face
- ✅ Compare provider performance (auto vs explicit selection)
- ✅ Implement failover and timeout strategies

### Quick Start

1. Read [01 Hugging Face Overview](01_huggingface_overview.md)
2. Review [02 Core Concepts](02_core_concepts.md)
3. Setup authentication following [03 Authentication & Security](03_authentication_security.md)
4. Complete [Image Generation Practice](01_image_generation_practice.ipynb)
5. Complete [Chat Inference Practice](02_chat_inference_practice.ipynb)

---

## Local Inference Endpoints (6 hours)

### 📚 Documentation

- **[05 Local Inference Overview](05_local_inference_overview.md)** - Introduction and requirements
- **[06 Inference Engines](06_inference_engines.md)** - Complete guide to Ollama and vLLM

### 💻 Jupyter Notebooks (Coding Practice)

- **[Ollama Practice](05_ollama_practice.ipynb)** - Install, configure, and test Ollama

### Learning Objectives

- ✅ Install and operate Ollama via CLI (pull/run/list/serve)
- ✅ Use Ollama's REST and OpenAI-compatible APIs
- ✅ Install and run vLLM in offline and service modes
- ✅ Configure vLLM's OpenAI-compatible server
- ✅ Compare throughput and latency between Ollama and vLLM
- ✅ Understand memory and hardware considerations

### Quick Start

1. Read [05 Local Inference Overview](05_local_inference_overview.md)
2. Study [06 Inference Engines](06_inference_engines.md) guide
3. Complete [Ollama Practice](05_ollama_practice.ipynb)
4. Experiment with vLLM deployment
5. Run performance benchmarks

---

## Assessment Checkpoints

### Hugging Face Platform
- ✅ Authenticate without exposing tokens in code
- ✅ Perform image and chat inference via Hugging Face providers
- ✅ Measure and compare latency/stability for different provider strategies
- ✅ Implement failover and error handling

### Local Inference Endpoints
- ✅ Install, start, and query both Ollama and vLLM
- ✅ Use Python clients for chat/completions against local endpoints
- ✅ Measure throughput (tokens/sec) and latency
- ✅ Articulate performance differences and use cases

---

## Common Pitfalls & Tips

### Hugging Face Platform
- ⚠️ **Token Security**: Never hardcode tokens; use environment variables
- ⚠️ **Timeouts**: Start conservative and implement exponential backoff
- ⚠️ **Provider Selection**: Different providers have different characteristics
- ⚠️ **Rate Limiting**: Implement delays between requests

### Local Inference Endpoints
- ⚠️ **Port Conflicts**: Ensure services use different ports (Ollama: 11434, vLLM: 8000)
- ⚠️ **Model Size**: Choose models that fit in available VRAM/RAM
- ⚠️ **Resource Monitoring**: Track GPU/CPU usage during tests
- ⚠️ **API Compatibility**: Verify OpenAI-compatible payload formats

---

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- GPU: 8GB VRAM (for 7B models)
- Storage: 50GB free

### Recommended
- CPU: 8+ cores
- RAM: 32GB+
- GPU: 16GB+ VRAM (for 13B models)
- Storage: 100GB+ SSD

---

## Additional Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/api-inference)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Next Steps

After completing this chapter:
- Build a production chatbot application
- Implement load balancing for multiple GPUs
- Explore model quantization techniques
- Deploy inference services to production
