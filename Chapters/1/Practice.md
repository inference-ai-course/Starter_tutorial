# Chapter 2: Setting Up Your Development Environment

## Learning Objectives

- Create and configure a complete development environment using the tools from Chapter 1
- Set up a remote development workspace with proper security
- Establish version control for your projects
- Create your first reproducible Python environment
- Launch and configure Jupyter for data science work

---

## 2.1 Environment Setup Workflow

### Overview
This chapter guides you through setting up a complete, professional development environment that combines all the tools we discussed in Chapter 1. By the end, you'll have a fully functional remote development setup.

### What You'll Build
- ✅ Secure SSH connection to remote server
- ✅ Isolated Python environment with conda
- ✅ Version-controlled project repository
- ✅ Configured Jupyter workspace
- ✅ VS Code remote development setup

---

## 2.2 Step-by-Step Setup Guide

### Step 1: Verify Tool Installation

Before proceeding, ensure all tools from Chapter 1 are properly installed:

```bash
# Check SSH
ssh -V

# Check Git
git --version

# Check Conda
conda --version

# Check VS Code (should open the editor)
code --version
```

### Step 2: Generate and Configure SSH Keys

If you haven't already set up SSH keys in Chapter 1:

```bash
# Generate new SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Test connection to GitHub
ssh -T git@github.com
```

### Step 3: Create Your Project Directory Structure

```bash
# Create main projects directory
mkdir -p ~/projects/data-science-workspace
cd ~/projects/data-science-workspace

# Create subdirectories
mkdir -p {data,notebooks,scripts,docs,tests}
mkdir -p data/{raw,processed,external}
mkdir -p notebooks/{exploratory,final}

# Create README file
cat > README.md << 'EOF'
# Data Science Workspace

## Project Structure
- `data/` - Data files (raw, processed, external)
- `notebooks/` - Jupyter notebooks (exploratory, final)
- `scripts/` - Python scripts and modules
- `docs/` - Documentation
- `tests/` - Test files

## Setup Instructions
See Chapter 2 of the tutorial series for detailed setup.
EOF
```

### Step 4: Initialize Git Repository

```bash
# Initialize Git repository
git init

# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter
.ipynb_checkpoints/

# Data (uncomment if data should be excluded)
# data/raw/
# data/external/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Environment
.env
.conda
EOF

# Make initial commit
git add .
git commit -m "Initial project structure"
```

### Step 5: Create Conda Environment

```bash
# Create environment with data science packages
conda create -n data-science-env python=3.12

# Activate environment
conda activate data-science-env

# Install core packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyterlab

# Install additional useful packages
conda install requests beautifulsoup4 plotly

# Optional: Install development tools
pip install black flake8 pytest
```

### Step 6: Configure Jupyter Environment

```bash
# Install and register IPython kernel
python -m pip install ipykernel
python -m ipykernel install --user --name data-science-env --display-name "Python (Data Science)"

# Create Jupyter config
jupyter lab --generate-config

# Create custom startup script
mkdir -p ~/.jupyter/startup
cat > ~/.jupyter/startup/00-startup.py << 'EOF'
# Data Science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Data Science environment loaded!")
EOF
```

### Step 7: Set Up VS Code Configuration

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${env:CONDA_PREFIX}/envs/data-science-env/bin/python",
    "python.terminal.activateEnvironment": true,
    "jupyter.askForKernelRestart": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    },
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

Create `.vscode/extensions.json`:
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-python.flake8",
        "ms-python.black-formatter",
        "eamodio.gitlens"
    ]
}
```

---

## 2.3 Advanced Configuration

### Environment Variables Setup

Create `.env` file for project-specific variables:
```bash
# Data paths
DATA_RAW_PATH=./data/raw
DATA_PROCESSED_PATH=./data/processed

# API keys (never commit these)
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key

# Jupyter settings
JUPYTER_PORT=8888
```

### Git Configuration for Data Science

```bash
# Configure Git for better collaboration
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Set up Git LFS for large files (optional)
git lfs install
git lfs track "*.csv"
git lfs track "*.json"
git lfs track "*.parquet"
```

### Conda Environment Export

```bash
# Export environment for reproducibility
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml
```

---

## 2.4 Testing Your Setup

### Verification Script

Create `scripts/verify_setup.py`:
```python
#!/usr/bin/env python3
"""
Verify that your development environment is properly configured.
"""

import sys
import subprocess
import importlib
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

def check_git_config():
    """Check Git configuration."""
    try:
        result = subprocess.run(['git', 'config', '--list'], 
                              capture_output=True, text=True)
        if 'user.name' in result.stdout and 'user.email' in result.stdout:
            print("✅ Git configuration")
            return True
        else:
            print("❌ Git configuration incomplete")
            return False
    except Exception:
        print("❌ Git not available")
        return False

def main():
    """Run all checks."""
    print("🔍 Verifying development environment setup...\n")
    
    checks = [
        check_python_version(),
        check_package('numpy'),
        check_package('pandas'),
        check_package('matplotlib'),
        check_package('jupyter'),
        check_package('sklearn'),
        check_git_config(),
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Your development environment is ready!")
        return 0
    else:
        print("⚠️  Some issues detected. Please review the setup instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run the verification:
```bash
python scripts/verify_setup.py
```

---

## 2.5 Best Practices and Tips

### Project Organization
- Keep raw data immutable (never modify original files)
- Use notebooks for exploration, scripts for production
- Document everything in README files
- Use meaningful commit messages

### Environment Management
- Create separate environments for different projects
- Export environment specifications regularly
- Keep base environment minimal
- Use conda-forge channel for more packages

### Security Considerations
- Never commit sensitive data (API keys, passwords)
- Use SSH keys for all Git operations
- Keep your SSH private key secure with a passphrase
- Regularly update your packages

---

## 2.6 Common Issues and Solutions

### Issue: Conda environment not activating in VS Code
**Solution**: Set the Python interpreter path in VS Code settings or select it manually (Ctrl+Shift+P → "Python: Select Interpreter")

### Issue: Jupyter kernel not found
**Solution**: Ensure the kernel is registered with `python -m ipykernel install --user --name env_name`

### Issue: Git asks for username/password
**Solution**: Set up SSH key authentication and configure Git to use SSH URLs

### Issue: Package conflicts in conda
**Solution**: Create a fresh environment or use `conda-forge` channel: `conda config --add channels conda-forge`

---

## 2.7 Next Steps

Now that your development environment is set up, you're ready to:

1. **Start your first data science project**
2. **Connect to remote servers for computation**
3. **Collaborate with others using Git**
4. **Explore advanced Jupyter features**
5. **Set up continuous integration**

### Recommended Learning Path:
- Chapter 3: Basic Data Analysis with Pandas
- Chapter 4: Data Visualization with Matplotlib and Seaborn
- Chapter 5: Introduction to Machine Learning

---

## References and Resources

### Official Documentation
- [Conda Documentation](https://docs.conda.io/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Git Documentation](https://git-scm.com/doc)
- [VS Code Documentation](https://code.visualstudio.com/docs)

### Best Practices Guides
- [Git Best Practices](https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices)
- [Python Packaging Guide](https://packaging.python.org/)
- [Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)

### Community Resources
- [Stack Overflow](https://stackoverflow.com/questions/tagged/conda)
- [Reddit r/datascience](https://www.reddit.com/r/datascience/)
- [GitHub Community](https://github.community/)

### Tools and Extensions
- [Oh My Zsh](https://ohmyz.sh/) - Terminal enhancement
- [GitHub CLI](https://cli.github.com/) - Command-line GitHub tool
- [Docker](https://www.docker.com/) - Containerization platform
