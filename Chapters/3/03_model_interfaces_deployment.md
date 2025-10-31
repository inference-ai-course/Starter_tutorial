 # Part 3: Model Interfaces and Deployment

## Learning Objectives

- Install and operate local inference engines like Ollama and vLLM
- Use OpenAI-compatible interfaces with HuggingFace and other open platforms
- Compare throughput, latency, and resource usage across different deployment options
- Implement authentication, security, and monitoring for production deployments
- Build scalable AI applications with proper error handling and failover mechanisms

## 3.1 Understanding Model Interfaces

### OpenAI-Compatible API Standard

The OpenAI API has become the de facto standard for language model interactions. Most modern inference engines support this format, enabling code portability across different providers.

#### Core API Structure

```python
# Universal client pattern that works with multiple providers
from openai import OpenAI

# OpenAI
client = OpenAI(api_key="your-openai-key")

# HuggingFace Inference API
client = OpenAI(
    api_key="your-hf-token",
    base_url="https://api-inference.huggingface.co/v1"
)

# Local Ollama
client = OpenAI(
    api_key="ollama",  # Ollama doesn't require authentication
    base_url="http://localhost:11434/v1"
)

# Local vLLM
client = OpenAI(
    api_key="vllm",  # vLLM doesn't require authentication
    base_url="http://localhost:8000/v1"
)
```

#### Standard Request Format

```python
def create_chat_completion(client, messages, model="gpt-3.5-turbo", **kwargs):
    """Create chat completion with universal parameters."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get('temperature', 0.7),
        max_tokens=kwargs.get('max_tokens', 150),
        top_p=kwargs.get('top_p', 1.0),
        frequency_penalty=kwargs.get('frequency_penalty', 0),
        presence_penalty=kwargs.get('presence_penalty', 0),
        stop=kwargs.get('stop', None),
        stream=kwargs.get('stream', False)
    )
    return response
```

## 3.2 HuggingFace Inference Providers

### HuggingFace Inference API

HuggingFace provides a unified interface for accessing thousands of models with built-in optimization and scaling.

#### Basic Usage

```python
from huggingface_hub import InferenceClient

# Initialize client
client = InferenceClient(
    token="your-hf-token",
    model="microsoft/DialoGPT-medium"
)

# Text generation
response = client.text_generation(
    "Hello, how are you?",
    max_new_tokens=50,
    temperature=0.7
)

# Chat completion (OpenAI-compatible)
response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=50,
    temperature=0.7
)
```

### Inference Providers and Failover

```python
class HuggingFaceProviderManager:
    def __init__(self, token, preferred_providers=None):
        self.token = token
        self.preferred_providers = preferred_providers or ["auto"]
        self.client = InferenceClient(token=token)
    
    def chat_completion_with_failover(self, messages, model, max_retries=3):
        """Attempt chat completion with provider failover."""
        for attempt in range(max_retries):
            try:
                # Try with specified providers
                for provider in self.preferred_providers:
                    try:
                        response = self.client.chat_completion(
                            messages=messages,
                            model=model,
                            provider=provider,
                            max_tokens=150
                        )
                        return response
                    except Exception as e:
                        print(f"Provider {provider} failed: {e}")
                        continue
                
                # Fallback to auto selection
                response = self.client.chat_completion(
                    messages=messages,
                    model=model,
                    provider="auto",
                    max_tokens=150
                )
                return response
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All providers failed")
```

### Model Selection and Optimization

```python
class ModelOptimizer:
    def __init__(self, client):
        self.client = client
        self.model_performance_cache = {}
    
    def select_optimal_model(self, task_type, requirements):
        """Select the best model based on task and requirements."""
        model_candidates = self.get_model_candidates(task_type)
        
        for model in model_candidates:
            if self.meets_requirements(model, requirements):
                return model
        
        # Fallback to default
        return self.get_default_model(task_type)
    
    def get_model_candidates(self, task_type):
        """Get list of suitable models for task type."""
        models = {
            'chat': [
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
                "google/flan-t5-base"
            ],
            'code': [
                "codellama/CodeLlama-7b-Instruct-hf",
                "WizardLM/WizardCoder-15B-V1.0",
                "Salesforce/codegen-350M-mono"
            ],
            'summarization': [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "microsoft/DialoGPT-medium"
            ]
        }
        return models.get(task_type, models['chat'])
    
    def benchmark_model(self, model, test_prompts):
        """Benchmark model performance."""
        results = {
            'model': model,
            'latency': [],
            'token_count': [],
            'quality_score': []
        }
        
        for prompt in test_prompts:
            start_time = time.time()
            
            response = self.client.text_generation(
                prompt,
                model=model,
                max_new_tokens=100
            )
            
            latency = time.time() - start_time
            token_count = len(response.split())
            
            results['latency'].append(latency)
            results['token_count'].append(token_count)
            results['quality_score'].append(self.assess_quality(response))
        
        return self.summarize_benchmark(results)
```

## 3.3 Local Inference with Ollama

### Ollama Installation and Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull models
ollama pull llama2
ollama pull mistral
ollama pull codellama

# List available models
ollama list
```

### Ollama API Integration

```python
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_chat = f"{base_url}/api/chat"
        self.api_tags = f"{base_url}/api/tags"
    
    def list_models(self):
        """List available models."""
        response = requests.get(self.api_tags)
        return response.json()['models']
    
    def generate_text(self, model, prompt, options=None):
        """Generate text using Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {}
        }
        
        response = requests.post(
            self.api_generate,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def chat_completion(self, model, messages, options=None):
        """Chat completion with message history."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {}
        }
        
        response = requests.post(
            self.api_chat,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def openai_compatible_chat(self, model, messages, **kwargs):
        """OpenAI-compatible chat interface."""
        # Convert OpenAI format to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        options = {
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.9),
            "top_k": kwargs.get('top_k', 40),
            "repeat_penalty": kwargs.get('frequency_penalty', 1.1)
        }
        
        response = self.chat_completion(model, ollama_messages, options)
        
        # Convert back to OpenAI format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response['message']['content']
                },
                "finish_reason": "stop"
            }],
            "model": model,
            "usage": {
                "prompt_tokens": 0,  # Ollama doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
```

### Ollama Performance Optimization

```python
class OllamaOptimizer:
    def __init__(self, client):
        self.client = client
    
    def optimize_model_loading(self, model_name, gpu_layers=None):
        """Optimize model loading for specific hardware."""
        options = {
            "num_gpu": gpu_layers or self.detect_optimal_gpu_layers(model_name),
            "num_thread": self.get_optimal_thread_count(),
            "context_length": 4096  # Adjust based on available memory
        }
        
        return options
    
    def detect_optimal_gpu_layers(self, model_name):
        """Detect optimal number of GPU layers based on available VRAM."""
        # This is a simplified example - in practice, you'd detect actual GPU memory
        import psutil
        import subprocess
        
        try:
            # Try to detect GPU memory (NVIDIA)
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            vram_mb = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Rough estimation - actual layers depend on model size
            if vram_mb > 8000:  # 8GB+ VRAM
                return 35
            elif vram_mb > 4000:  # 4GB+ VRAM
                return 20
            else:
                return 10
                
        except:
            return 0  # CPU only
    
    def get_optimal_thread_count(self):
        """Get optimal thread count for CPU inference."""
        return min(psutil.cpu_count(logical=False), 8)  # Cap at 8 physical cores
    
    def benchmark_ollama_model(self, model_name, test_prompts):
        """Benchmark Ollama model performance."""
        results = {
            'model': model_name,
            'total_latency': 0,
            'token_throughput': [],
            'memory_usage': [],
            'responses': []
        }
        
        for prompt in test_prompts:
            start_time = time.time()
            
            response = self.client.generate_text(model_name, prompt)
            
            latency = time.time() - start_time
            token_count = len(response.split())
            
            results['total_latency'] += latency
            results['token_throughput'].append(token_count / latency)
            results['responses'].append(response)
            
            # Monitor memory usage if possible
            memory_mb = self.get_memory_usage()
            if memory_mb:
                results['memory_usage'].append(memory_mb)
        
        return self.summarize_benchmark(results)
```

## 3.4 High-Performance Inference with vLLM

### vLLM Installation and Setup

```bash
# Install vLLM with CUDA support
pip install vllm

# For specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000
```

### vLLM Client Integration

```python
class VLLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = OpenAI(
            api_key="vllm",  # vLLM doesn't require authentication
            base_url=f"{base_url}/v1"
        )
    
    def generate_with_vllm(self, prompt, model, **kwargs):
        """Generate text using vLLM with optimized parameters."""
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 150),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', -1),
            frequency_penalty=kwargs.get('frequency_penalty', 0),
            presence_penalty=kwargs.get('presence_penalty', 0),
            stop=kwargs.get('stop', None),
            stream=kwargs.get('stream', False),
            best_of=kwargs.get('best_of', 1),
            use_beam_search=kwargs.get('use_beam_search', False)
        )
        
        return response
    
    def chat_with_vllm(self, messages, model, **kwargs):
        """Chat completion with vLLM optimization."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 150),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            frequency_penalty=kwargs.get('frequency_penalty', 0),
            presence_penalty=kwargs.get('presence_penalty', 0),
            stop=kwargs.get('stop', None),
            stream=kwargs.get('stream', False)
        )
        
        return response
    
    def benchmark_vllm_performance(self, test_prompts, model):
        """Benchmark vLLM performance metrics."""
        results = {
            'model': model,
            'total_requests': len(test_prompts),
            'successful_requests': 0,
            'total_latency': 0,
            'throughput_per_second': [],
            'token_counts': []
        }
        
        start_time = time.time()
        
        for prompt in test_prompts:
            request_start = time.time()
            
            try:
                response = self.generate_with_vllm(prompt, model, max_tokens=100)
                results['successful_requests'] += 1
                
                latency = time.time() - request_start
                results['total_latency'] += latency
                
                # Count tokens (approximate)
                token_count = len(response.choices[0].text.split())
                results['token_counts'].append(token_count)
                results['throughput_per_second'].append(token_count / latency)
                \n            except Exception as e:
n                print(f"Request failed: {e}")
n        \n        total_time = time.time() - start_time\n        results['total_time'] = total_time\n        results['overall_throughput'] = results['successful_requests'] / total_time\n        \n        return results
```

### vLLM Advanced Configuration

```python
class VLLMConfig:
    def __init__(self):
        self.default_config = {
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.9,
            'max_model_len': 4096,
            'dtype': 'auto',
            'quantization': None,
            'seed': 0,
            'swap_space': 4,
            'enforce_eager': False,
            'max_context_len_to_capture': 8192,
            'block_size': 16,
            'max_num_batched_tokens': 4096,
            'max_num_seqs': 256,
            'max_paddings': 256
        }
    
    def get_optimized_config(self, model_name, hardware_specs):
        """Get optimized configuration based on hardware."""
        config = self.default_config.copy()
        
        # Adjust based on GPU memory
        if hardware_specs.get('gpu_memory_gb', 0) > 24:
            config['gpu_memory_utilization'] = 0.95
            config['max_model_len'] = 8192
            config['tensor_parallel_size'] = 2
        elif hardware_specs.get('gpu_memory_gb', 0) > 12:
            config['gpu_memory_utilization'] = 0.9
            config['max_model_len'] = 4096
        else:
            config['gpu_memory_utilization'] = 0.8
            config['max_model_len'] = 2048
        
        # Adjust based on CPU cores
        if hardware_specs.get('cpu_cores', 0) > 16:
            config['max_num_seqs'] = 512
            config['max_num_batched_tokens'] = 8192
        
        return config
    
    def create_deployment_script(self, model_name, config, port=8000):
        """Create deployment script for vLLM server."""
        script = f"""#!/bin/bash
# vLLM deployment script for {model_name}

python -m vllm.entrypoints.openai.api_server \\
    --model {model_name} \\
    --tensor-parallel-size {config['tensor_parallel_size']} \\
    --gpu-memory-utilization {config['gpu_memory_utilization']} \\
    --max-model-len {config['max_model_len']} \\
    --dtype {config['dtype']} \\
    --port {port} \\
    --host 0.0.0.0 \\
    --allow-credentials \\
    --allowed-origins ["*"] \\
    --api-key vllm-secret-key

echo "vLLM server started on port {port}"
"""
        return script
```

## 3.5 Deployment Architecture and Best Practices

### Production Deployment Patterns

```python
class ProductionDeployment:
    def __init__(self, config):
        self.config = config
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker()
        self.monitoring = MonitoringSystem()
    
    def setup_production_environment(self):
        """Setup production-ready environment with proper architecture."""
        deployment = {
            'load_balancer': self.configure_load_balancer(),
            'inference_servers': self.setup_inference_cluster(),
            'monitoring': self.setup_monitoring(),
            'security': self.configure_security(),
            'backup': self.setup_backup_systems(),
            'scaling': self.configure_auto_scaling()
        }
        return deployment
    
    def configure_load_balancer(self):
        """Configure load balancer for high availability."""
        return {
            'algorithm': 'round_robin',
            'health_checks': {
                'interval': 30,
                'timeout': 5,
                'healthy_threshold': 2,
                'unhealthy_threshold': 3
            },
            'ssl_termination': True,
            'rate_limiting': {
                'requests_per_minute': 1000,
                'burst_size': 100
            }
        }
    
    def setup_inference_cluster(self):
        """Setup cluster of inference servers."""
        servers = []
        
        # Primary servers
        for i in range(self.config.get('primary_servers', 2)):
            server = {
                'id': f'primary_{i}',
                'type': 'primary',
                'model': self.config['model_name'],
                'engine': self.config['inference_engine'],
                'resources': {
                    'gpu_memory': self.config.get('gpu_memory', '8GB'),
                    'cpu_cores': self.config.get('cpu_cores', 4),
                    'ram': self.config.get('ram', '16GB')
                },
                'health_check': {
                    'endpoint': '/health',
                    'interval': 10
                }
            }
            servers.append(server)
        
        # Backup servers
        for i in range(self.config.get('backup_servers', 1)):
            server = {
                'id': f'backup_{i}',
                'type': 'backup',
                'model': self.config['model_name'],
                'engine': self.config['inference_engine'],
                'resources': {
                    'gpu_memory': self.config.get('backup_gpu_memory', '4GB'),
                    'cpu_cores': self.config.get('backup_cpu_cores', 2),
                    'ram': self.config.get('backup_ram', '8GB')
                }
            }
            servers.append(server)
        
        return servers
    
    def setup_monitoring(self):
        """Setup comprehensive monitoring system."""
        return {
            'metrics': {
                'latency_p50': {'threshold': 1000, 'unit': 'ms'},
                'latency_p95': {'threshold': 2000, 'unit': 'ms'},
                'latency_p99': {'threshold': 5000, 'unit': 'ms'},
                'error_rate': {'threshold': 0.01, 'unit': 'percentage'},
                'throughput': {'threshold': 100, 'unit': 'requests_per_second'},
                'gpu_utilization': {'threshold': 90, 'unit': 'percentage'},
                'memory_usage': {'threshold': 85, 'unit': 'percentage'}
            },
            'alerts': {
                'email': self.config.get('alert_email'),
                'slack': self.config.get('alert_slack'),
                'pagerduty': self.config.get('alert_pagerduty')
            },
            'dashboards': {
                'grafana': True,
                'prometheus': True,
                'custom_metrics': True
            }
        }
    
    def configure_security(self):
        """Configure security measures."""
        return {
            'authentication': {
                'type': 'api_key',
                'rate_limiting': {
                    'per_user': 100,  # requests per minute
                    'per_ip': 500     # requests per minute
                }
            },
            'encryption': {
                'tls': True,
                'certificates': 'lets_encrypt'
            },
            'network': {
                'firewall_rules': [
                    'ALLOW 443/tcp',  # HTTPS
                    'ALLOW 80/tcp',   # HTTP (redirects to HTTPS)
                    'DENY ALL'
                ],
                'vpc_isolation': True
            },
            'data_protection': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'audit_logging': True
            }
        }
    
    def setup_backup_systems(self):
        """Setup backup and disaster recovery."""
        return {
            'model_checkpoints': {
                'frequency': 'daily',
                'retention': '30_days',
                'location': 's3://model-backups/'
            },
            'configuration_backup': {
                'frequency': 'hourly',
                'retention': '7_days',
                'location': 'git_repository'
            },
            'disaster_recovery': {
                'rto': '15_minutes',  # Recovery Time Objective
                'rpo': '5_minutes',   # Recovery Point Objective
                'backup_site': 'different_region'
            }
        }
    
    def configure_auto_scaling(self):
        """Configure auto-scaling policies."""
        return {
            'horizontal_scaling': {
                'min_instances': 2,
                'max_instances': 10,
                'target_cpu_utilization': 70,
                'scale_up_cooldown': 300,  # seconds
                'scale_down_cooldown': 600  # seconds
            },
            'vertical_scaling': {
                'enabled': True,
                'max_gpu_memory': '24GB',
                'max_cpu_cores': 16
            },
            'predictive_scaling': {
                'enabled': True,
                'look_ahead_time': '1_hour',
                'historical_data_days': 7
            }
        }

class SecurityManager:
    def __init__(self, config):
        self.config = config
        self.api_keys = {}
        self.rate_limiter = RateLimiter()
    
    def generate_api_key(self, user_id, permissions):
        """Generate secure API key with permissions."""
        import secrets
        import hashlib
        
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        return key
    
    def validate_api_key(self, api_key):
        """Validate API key and check permissions."""
        import hashlib
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return False, "Invalid API key"
        
        key_info = self.api_keys[key_hash]
        
        # Check if key is still valid
        if self.is_key_expired(key_info):
            return False, "API key expired"
        
        # Update usage
        key_info['last_used'] = datetime.now().isoformat()
        key_info['usage_count'] += 1
        
        return True, key_info['permissions']
    
    def is_key_expired(self, key_info):
        """Check if API key has expired."""
        # Implement expiration logic based on your policy
        created_at = datetime.fromisoformat(key_info['created_at'])
        expiration_days = self.config.get('api_key_expiration_days', 90)
        
        return (datetime.now() - created_at).days > expiration_days

class MonitoringSystem:
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.alerts = []
    
    def record_metric(self, metric_name, value, tags=None):
        """Record a metric with optional tags."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'tags': tags or {}
        })
    
    def check_alerts(self):
        """Check if any metrics exceed thresholds."""
        thresholds = self.config.get('thresholds', {})
        
        for metric_name, data in self.metrics.items():
            if metric_name in thresholds:
                latest_value = data[-1]['value']
                threshold = thresholds[metric_name]
                
                if latest_value > threshold['max'] or latest_value < threshold['min']:
                    self.trigger_alert(metric_name, latest_value, threshold)
    
    def trigger_alert(self, metric_name, value, threshold):
        """Trigger alert for metric threshold breach."""
        alert = {
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'severity': self.calculate_severity(value, threshold)
        }
        
        self.alerts.append(alert)
        self.send_notification(alert)
    
    def calculate_severity(self, value, threshold):
        """Calculate alert severity based on deviation from threshold."""
        if 'max' in threshold and value > threshold['max']:
            deviation = (value - threshold['max']) / threshold['max']
        elif 'min' in threshold and value < threshold['min']:
            deviation = (threshold['min'] - value) / threshold['min']
        else:
            return 'info'
        
        if deviation > 0.5:
            return 'critical'
        elif deviation > 0.2:
            return 'warning'
        else:
            return 'info'
    
    def send_notification(self, alert):
        """Send alert notification through configured channels."""
        # Implement notification logic (email, Slack, PagerDuty, etc.)
        print(f"ALERT: {alert['metric']} = {alert['value']} (severity: {alert['severity']})")

# Example usage
def deploy_production_system():
    """Example of deploying a production-ready AI inference system."""
    
    config = {
        'model_name': 'microsoft/DialoGPT-medium',
        'inference_engine': 'vllm',
        'primary_servers': 3,
        'backup_servers': 2,
        'gpu_memory': '16GB',
        'cpu_cores': 8,
        'ram': '32GB',
        'alert_email': 'ops@company.com',
        'alert_slack': '#alerts'
    }
    
    deployment = ProductionDeployment(config)
    environment = deployment.setup_production_environment()
    
    print("Production environment configured:")
    print(f"- Load balancer: {environment['load_balancer']['algorithm']}")
    print(f"- Inference servers: {len(environment['inference_servers'])}")
    print(f"- Monitoring: {len(environment['monitoring']['metrics'])} metrics")
    print(f"- Security: {environment['security']['authentication']['type']}")
    
    return environment

if __name__ == "__main__":
    environment = deploy_production_system()
```

## 3.6 Performance Comparison and Benchmarking

### Comprehensive Benchmarking Framework

```python
class InferenceBenchmark:
    def __init__(self):
        self.results = {}
        self.test_prompts = self.load_test_prompts()
    
    def load_test_prompts(self):
        """Load diverse test prompts for benchmarking."""
        return [
            "Write a short story about artificial intelligence.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the process of photosynthesis.",
            "What are the key principles of software engineering?",
            "Explain the difference between SQL and NoSQL databases.",
            "How do neural networks learn?",
            "What is the importance of data structures?",
            "Describe a sustainable city of the future."
        ]
    
    def benchmark_all_engines(self):
        """Benchmark all available inference engines."""
        engines = [
            ('openai', self.benchmark_openai),
            ('huggingface', self.benchmark_huggingface),
            ('ollama', self.benchmark_ollama),
            ('vllm', self.benchmark_vllm)
        ]
        
        for engine_name, benchmark_func in engines:
            try:
                print(f"Benchmarking {engine_name}...")
                results = benchmark_func()
                self.results[engine_name] = results
            except Exception as e:
                print(f"Failed to benchmark {engine_name}: {e}")
                self.results[engine_name] = {'error': str(e)}
        
        return self.generate_comparison_report()
    
    def benchmark_openai(self):
        """Benchmark OpenAI API."""
        from openai import OpenAI
        client = OpenAI()
        
        results = {
            'latencies': [],
            'token_counts': [],
            'throughput': [],
            'cost_estimates': []
        }
        
        for prompt in self.test_prompts:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            
            latency = time.time() - start_time
            token_count = response.usage.total_tokens
            
            results['latencies'].append(latency)
            results['token_counts'].append(token_count)
            results['throughput'].append(token_count / latency)
            results['cost_estimates'].append(token_count * 0.002 / 1000)  # Approximate cost
        
        return self.summarize_results(results)
    
    def benchmark_huggingface(self):
        """Benchmark HuggingFace Inference API."""
        from huggingface_hub import InferenceClient
        client = InferenceClient()
        
        results = {
            'latencies': [],
            'token_counts': [],
            'throughput': []
        }
        
        for prompt in self.test_prompts:
            start_time = time.time()
            
            response = client.text_generation(
                prompt,
                model="microsoft/DialoGPT-medium",
                max_new_tokens=150
            )
            
            latency = time.time() - start_time
            token_count = len(response.split())
            
            results['latencies'].append(latency)
            results['token_counts'].append(token_count)
            results['throughput'].append(token_count / latency)
        
        return self.summarize_results(results)
    
    def benchmark_ollama(self):
        """Benchmark Ollama local inference."""
        client = OllamaClient()
        
        results = {
            'latencies': [],
            'token_counts': [],
            'throughput': [],
            'memory_usage': []
        }
        
        for prompt in self.test_prompts:
            start_time = time.time()
            
            response = client.generate_text("llama2", prompt)
            
            latency = time.time() - start_time
            token_count = len(response.split())
            
            results['latencies'].append(latency)
            results['token_counts'].append(token_count)
            results['throughput'].append(token_count / latency)
            results['memory_usage'].append(self.get_memory_usage())
        
        return self.summarize_results(results)
    
    def benchmark_vllm(self):
        """Benchmark vLLM local inference."""
        client = VLLMClient()
        
        results = {
            'latencies': [],
            'token_counts': [],
            'throughput': [],
            'memory_usage': []
        }
        
        for prompt in self.test_prompts:
            start_time = time.time()
            
            response = client.generate_with_vllm(
                prompt,
                "microsoft/DialoGPT-medium",
                max_tokens=150
            )
            
            latency = time.time() - start_time
            token_count = len(response.choices[0].text.split())
            
            results['latencies'].append(latency)
            results['token_counts'].append(token_count)
            results['throughput'].append(token_count / latency)
            results['memory_usage'].append(self.get_memory_usage())
        
        return self.summarize_results(results)
    
    def summarize_results(self, results):
        """Summarize benchmarking results."""
        summary = {}
        
        for metric, values in results.items():
            if values and isinstance(values[0], (int, float)):
                summary[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
