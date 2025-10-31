# Part 5: Advanced Environment Topics

## Introduction

This section covers advanced topics in environment management, including mixed package manager usage, Jupyter integration, secure handling of secrets, and containerization with Docker. These skills are essential for professional AI engineering workflows.

## Table of Contents
1. [Mixed Conda/Pip Usage](#mixed-conda-pip-usage)
2. [Jupyter Integration](#jupyter-integration)
3. [Environment Variables and Security](#environment-variables-and-security)
4. [Docker Containerization](#docker-containerization)
5. [Performance Optimization](#performance-optimization)
6. [Practical Exercises](#practical-exercises)

---

## Mixed Conda/Pip Usage

### When to Use Mixed Package Managers

```bash
# Use Conda when:
# - Package is available in conda-forge
# - You need binary packages (faster installation)
# - Working with scientific computing libraries

# Use pip when:
# - Package is not available in conda
# - You need the latest version
# - Working with pure Python packages
```

### Best Practices for Mixed Usage

```bash
# 1. Create base environment with conda
conda create -n mixed-env python=3.10
conda activate mixed-env

# 2. Install conda packages first
conda install -c conda-forge numpy pandas matplotlib scikit-learn

# 3. Install pip packages that aren't in conda
pip install transformers huggingface_hub openai

# 4. Document what came from where
echo "# Environment Sources" > package_sources.txt
echo "## Conda Packages:" >> package_sources.txt
conda list >> package_sources.txt
echo "## Pip Packages:" >> package_sources.txt
pip list >> package_sources.txt
```

### Avoiding Conflicts

```bash
# Create a constraint file to prevent conflicts
cat > constraints.txt << 'EOF'
# Prevent pip from upgrading conda-managed packages
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
EOF

# Install pip packages with constraints
pip install -c constraints.txt some-package

# Or use --no-deps to avoid dependency resolution
pip install --no-deps package-name
```

### Creating Hybrid environment.yml Files

```yaml
name: hybrid-ai-env
channels:
  - conda-forge
  - pytorch
  - defaults

dependencies:
  # Core packages from conda
  - python=3.10
  - numpy=1.24
  - pandas=2.0
  - matplotlib=3.7
  - scikit-learn=1.3
  - pytorch=2.0
  - torchvision=0.15
  
  # Jupyter ecosystem
  - jupyter=1.0
  - jupyterlab=4.0
  - ipykernel=6.0
  
  # Development tools
  - black=23.0
  - pytest=7.4
  
  # Pip for packages not in conda
  - pip
  
  # Pip packages
  - pip:
    - transformers>=4.40
    - huggingface_hub>=0.22
    - datasets>=2.14
    - evaluate>=0.4
    - accelerate>=0.20
    - wandb>=0.15
    - openai>=1.0
    - tiktoken>=0.5
```

---

## Jupyter Integration

### Registering Kernels

```bash
# Install ipykernel in your environment
conda install -c conda-forge ipykernel

# Register kernel for Jupyter
python -m ipykernel install --user --name hybrid-ai-env --display-name "Python (AI Environment)"

# List available kernels
jupyter kernelspec list

# Remove a kernel
jupyter kernelspec uninstall hybrid-ai-env
```

### Multiple Kernel Management

```python
# Create a script to manage multiple kernels
# save as manage_kernels.py
import subprocess
import json

def list_kernels():
    """List all Jupyter kernels."""
    result = subprocess.run(['jupyter', 'kernelspec', 'list', '--json'], 
                          capture_output=True, text=True)
    return json.loads(result.stdout)

def install_kernel(env_name, display_name=None):
    """Install a kernel for a conda environment."""
    if display_name is None:
        display_name = f"Python ({env_name})"
    
    subprocess.run([
        'conda', 'run', '-n', env_name,
        'python', '-m', 'ipykernel', 'install',
        '--user', '--name', env_name,
        '--display-name', display_name
    ])

def remove_kernel(kernel_name):
    """Remove a Jupyter kernel."""
    subprocess.run(['jupyter', 'kernelspec', 'remove', '-f', kernel_name])

# Example usage
if __name__ == "__main__":
    kernels = list_kernels()
    print("Available kernels:")
    for kernel, path in kernels['kernelspecs'].items():
        print(f"  {kernel}: {path}")
```

### JupyterLab Extensions

```bash
# Install JupyterLab with extensions
conda install -c conda-forge jupyterlab

# Useful extensions for AI development
conda install -c conda-forge jupyterlab-git
pip install jupyterlab-system-monitor
pip install jupyterlab-lsp
conda install -c conda-forge python-lsp-server

# Enable extensions
jupyter server extension enable jupyterlab_git
jupyter server extension enable jupyterlab_system_monitor
```

### Custom Jupyter Configuration

Create `~/.jupyter/jupyter_lab_config.py`:
```python
c = get_config()

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Security
c.ServerApp.token = ''
c.ServerApp.password = ''

# Kernel configuration
c.KernelManager.autorestart = True
c.KernelManager.cull_idle_timeout = 3600  # 1 hour
c.KernelManager.cull_interval = 300      # 5 minutes

# Resource limits
c.ResourceUseDisplay.track_cpu_percent = True
c.ResourceUseDisplay.cpu_limit = 2.0
c.ResourceUseDisplay.mem_limit = 8589934592  # 8GB in bytes

# Extension settings
c.GitCommitPush.git_dir = '.'
```

---

## Environment Variables and Security

### Secure Environment Variable Management

Create a `.env` file in your project directory:
```
# .env
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_key_here
DATABASE_URL=your_database_url_here
API_BASE_URL=https://api.example.com
```

Load environment variables securely:
```python
# load_env.py
import os
import warnings
from pathlib import Path
from typing import Optional

class EnvironmentManager:
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.load_env_file()
    
    def load_env_file(self):
        """Load environment variables from .env file."""
        if not self.env_file.exists():
            warnings.warn(f"No {self.env_file} file found. Some features may not work.")
            return
        
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Get environment variable with optional default."""
        value = os.environ.get(key, default)
