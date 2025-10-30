# Jupyter: Interactive Computing Environment

## Overview

Jupyter Notebook/Lab provides interactive computing with code cells, rich outputs, and narrative text. It's the standard tool for data science, research, and exploratory programming.

---

## What is Jupyter?

**Jupyter Architecture:**
- **Multi-language support**: Runs code through language-specific "kernels"
- **Interactive cells**: Execute code in segments with immediate feedback
- **Rich outputs**: Display plots, tables, and multimedia
- **Documentation**: Combine code with explanatory text and equations

**Common Kernels:**
- Python (IPyKernel)
- R
- Julia
- And many more...

---

## Core Concepts

### IPyKernel
IPyKernel (IPython kernel) is the Python implementation for Jupyter. It:
- Executes Python cells
- Communicates with notebook frontend via Jupyter's protocol
- Provides Python-specific features and magics

### Remote Access
When running on remote servers, access Jupyter securely via:
- SSH port forwarding
- VS Code's port forwarding
- Direct browser access

---

## Installation and Setup

### Step 1: Create and Activate Environment

Choose your preferred method:

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows: .venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n myenv python=3.12
conda activate myenv
```

### Step 2: Install Packages

**Recommended approach** (ensures install goes to active Python):
```bash
python -m pip install package_name
```

**In notebooks**, use kernel-specific magic:
```python
%pip install package_name
```

### Step 3: Install and Register Jupyter Kernel

```bash
# Install IPyKernel
python -m pip install ipykernel

# Register kernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

**Notes:**
- You don't need full jupyter package, only ipykernel
- List available kernels: `jupyter kernelspec list`

### Step 4: Launch Jupyter and Select Kernel

**Jupyter Notebook:**
- Menu: Kernel → Change kernel → Select "Python (myenv)"

**JupyterLab:**
- Click kernel name in top-right → Select "Python (myenv)"

---

## Launching Jupyter

### Local Access
```bash
jupyter lab
# or
jupyter notebook
```

### Remote Access
1. Run on remote server: `jupyter lab --no-browser --port=8888`
2. Use SSH port forwarding or VS Code port forwarding
3. Access via browser at forwarded port

---

## Troubleshooting

### Common Issues and Solutions

#### "Installed but cannot import"
**Problem**: Installed packages in wrong environment
**Solution**: Use `%pip install` or `!{sys.executable} -m pip install` in notebooks

#### Packages not recognized after install
**Problem**: Kernel needs restart
**Solution**: Kernel → Restart (picks up new packages)

#### Wrong Python environment
**Problem**: Kernel using different Python
**Solution**: Check and verify:
```python
import sys
print(sys.executable)
```

---

## Best Practices

### Environment Management
- Always use environments for projects
- Register each environment as separate kernel
- Name kernels descriptively

### Package Installation
- Use `%pip install` in notebooks
- Install packages in same environment as kernel
- Restart kernel after major package changes

### Notebook Organization
- Use clear, descriptive cell comments
- Separate code into logical sections
- Include markdown cells for explanations
- Save and checkpoint regularly

---

## Advanced Features

### Jupyter Magics
```python
# Timing
%time command

# Multiple commands
%%time
command1
command2

# Shell commands
!ls -la
```

### Extensions
- JupyterLab extensions for enhanced functionality
- Notebook extensions for additional features
- Custom themes and layouts

---

## Integration with Development Workflow

### Typical Workflow:
1. **Create environment** with required packages
2. **Register kernel** for that environment
3. **Launch Jupyter** and select appropriate kernel
4. **Develop interactively** with immediate feedback
5. **Export results** or convert to scripts
6. **Version control** notebooks with Git

### VS Code Integration:
- Native Jupyter support in VS Code
- Remote development with Jupyter on servers
- Convert between notebooks and Python scripts
- Debug notebooks directly in VS Code
