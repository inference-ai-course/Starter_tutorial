Section 5: Hugging Face Platform/Library Basics (4 hours)

Learning Objectives


Understand what Hugging Face Inference Providers are, how to select providers, implement failover and timeouts, and manage authentication and billing. 
Use OpenAI-compatible interfaces with Hugging Face and connect to local endpoints via HTTP for unified client code patterns. 

Concepts and Explanations


Inference Providers: A set of backend providers behind a unified interface on Hugging Face; you can choose providers explicitly or allow provider="auto" for dynamic selection based on availability and performance. This supports failover and configurable timeouts to improve reliability. 
Authentication & Billing: Securely use tokens (e.g., HF tokens) and understand how usage can be billed per provider; design workflows that keep credentials out of source control (e.g., environment variables, secret stores). 
OpenAI-Compatible Interfaces: Many inference backends expose an OpenAI-style API, enabling common client libraries and payload schemas across cloud and local endpoints. 
Connecting to Local Endpoints via HTTP: Use HTTP clients (e.g., requests) or OpenAI-compatible SDKs configured with a base_url pointing to your local service. 

Suggested Teaching Flow


Lecture/Demo (60–75 min): Review provider selection strategies, authentication/token handling, and timeout/retry patterns. Show how OpenAI-compatible clients can talk to both Hugging Face providers and local endpoints with minimal code changes. 
Guided Practice (30–45 min): Students configure credentials and run a first cloud inference. 
Measurement & Comparison (30–45 min): Benchmark provider="auto" vs. explicitly selected providers and record latency/stability differences. 

Hands-On Lab (Step-by-Step)


Goal: Run one image generation and one chat inference using Hugging Face Inference Providers; compare provider="auto" vs explicit selection for latency and stability. 

Set credentials securely (e.g., export HF_TOKEN in shell; avoid hardcoding in notebooks). 
 
Image generation:
Use a Hugging Face client or HTTP POST to the model’s inference endpoint to generate an image from a text prompt.
Repeat the same prompt with provider="auto" and with a specific provider; record response time and any errors/timeouts. 
 
Chat inference:
Send a user message to a chat-capable model via the Hugging Face Inference Providers API.
Compare outputs and latency across provider="auto" and explicit provider. 
Logging & Metrics:
Record average latency (ms), success rate, and any failovers; summarize stability differences. 

Example Snippets (for illustration)


Python (image generation, pseudocode):
Using huggingface_hub.InferenceClient with token, run text-to-image; set provider="auto" vs provider="X"; measure time with time.perf_counter(). 
Python (chat inference, pseudocode):
Use client.chat or a generic POST to the provider’s chat endpoint; collect model output and timing; log exceptions to assess timeouts/failovers. 

Assessment Checkpoints


Student can authenticate without exposing tokens in code. 
Student can perform both image and chat inference via providers. 
Student can measure and compare latency/stability for provider="auto" vs explicit providers and explain trade-offs. 

Common Pitfalls & Tips


Token misconfiguration: Ensure environment variables are loaded in terminal/Notebook sessions. 
Timeouts too short: Start with conservative timeouts and implement retries/backoff. 
 
Provider mismatch: Some models perform differently across providers; treat provider selection as a tuning parameter. 

Section 6: Local Inference Endpoints — Ollama and vLLM (6 hours)

Learning Objectives


Install and operate Ollama via CLI (pull/run/list/serve), use its local REST and OpenAI-compatible service, and understand throughput/memory considerations. 
Install and run vLLM in offline/service modes, use its OpenAI-compatible server, and compare throughput/latency with Ollama. 

Concepts and Explanations


Ollama Overview: A local inference runner with a simple CLI and REST API, suitable for developer laptops and workstations. It can also expose OpenAI-compatible endpoints for standardized client usage. 
 
vLLM Overview: A high-performance inference engine/server with an OpenAI-compatible API; optimized for throughput and efficient memory usage, often favored for serving larger models or multi-user workloads. 
 
Throughput & Memory: Larger models need more VRAM/RAM; batch size, tensor parallelism, and KV cache tuning impact throughput and latency; measure under realistic loads. 
 

Suggested Teaching Flow


Lecture/Demo (60–90 min): Install Ollama and vLLM; explain CLI/server modes; start services; review endpoint URLs and basic authentication options; discuss hardware constraints and performance trade-offs. 
Guided Practice (60 min): Students run both endpoints locally; send chat/completions using Python clients. 
Benchmarking (60–90 min): Compare throughput and latency across endpoints under similar prompts and parameters; capture logs and resource usage. 

Hands-On Lab (Step-by-Step)


Goal: Run Ollama and vLLM locally; perform chat/completions via Python clients; compare throughput and latency across endpoints. 

Ollama installation & basic CLI:
Install Ollama; verify with ollama --version. 
Pull and run a model: ollama pull <model>; ollama run <model>; list with ollama list; start service with ollama serve. 
 
Test REST: POST to http://localhost:11434/api/chat or completions endpoint with a prompt; validate JSON response. 
 
vLLM setup:
Install vLLM and start the OpenAI-compatible server (e.g., python -m vllm.entrypoints.openai.api_server --model <model>). 
 
Confirm server health; note base_url (e.g., http://localhost:8000/v1). 
Python client tests:
Use OpenAI-compatible SDK with base_url set to the local endpoint; send chat/completions; capture response time and token throughput. 
 
Benchmark & Compare:
Run standardized prompts across Ollama and vLLM; record tokens/sec, average latency, and max memory usage; summarize trade-offs in a short write-up. 

Example Snippets (for illustration)


Ollama REST (Python requests):
POST a JSON payload with messages to /api/chat; parse streaming or full response; log timing. 
vLLM OpenAI-compatible client:
Configure client with base_url=http://localhost:8000/v1 and api_key="dummy"; call chat.completions.create; measure latency and tokens. 

Assessment Checkpoints


Student can install, start, and query both Ollama and vLLM. 
Student can use Python clients to perform chat/completions against local endpoints. 
Student can measure throughput/latency and articulate performance differences between Ollama and vLLM. 

Common Pitfalls & Tips


Port conflicts: Ensure services are not overlapping; change ports or stop existing services. 
Model size vs hardware: Pick models that fit device memory; monitor usage during tests. 
 
SDK/base_url mismatches: Verify you’re sending OpenAI-compatible payloads to endpoints that support that protocol. 