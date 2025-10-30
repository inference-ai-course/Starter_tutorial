# Chapter 1: Tool Preparation

## Learning Objectives

- Understand a modern workflow that combines Shell/SSH, VS Code (and Remote-SSH), Git, Conda, and Jupyter
- Gain confidence with basic command-line operations
- Complete a hands-on lab: install Miniconda, configure VS Code Remote-SSH, generate SSH keys, and connect to a remote host

---

## 1. Workflow Overview

This chapter covers the essential tools for modern development:
- **Shell/SSH**: Command-line interface and remote access
- **VS Code**: Code editor with remote development capabilities
- **Git**: Version control system
- **Conda**: Environment and package management
- **Jupyter**: Interactive computing environment

---

## 2. Basic Command-Line Usage

### 2.1 Concept: Shell and Terminal

**Shell** is a command-line interpreter that processes commands and outputs results. Shells can also serve as scripting languages for automation (e.g., Bash, Zsh, PowerShell).

**Terminal** is the program that hosts the shell session. The "command line" refers to the interface where commands are entered.

**Why use the terminal?**
- Efficient for repetitive tasks
- Automation via scripts
- Remote administration where no GUI is available

### 2.2 Navigation and File Operations

**Essential Commands:**
- `cd` - Changes directories
- `ls` - Lists directory contents (`-l` for long listing)
- `mkdir` - Creates directories
- `touch` - Creates files
- `cp` - Copies files
- `mv` - Moves/renames files
- `rm` - Removes files (`-r` for directories - use carefully)
- Wildcards like `*` help operate on multiple files

**Command Options/Flags:** Modify behavior (e.g., `ls -lah`). Use `man <command>` to see a command's manual page.

**Windows Users:** Prefer WSL, PowerShell, or Git Bash for Unix-like parity when following tutorials.

**Shell Features:**
- Shell vs terminal vs CLI distinctions
- Autocompletion/tab features improve speed and accuracy
- Piping and chaining commands

---

## 3. Related Topics

### [Git (Local + VS Code Integration)](Git.md)
Version control system for tracking code changes

### [SSH](SSH.md)
Secure remote access and file transfer

### [Conda (Environments and Packages)](Conda.md)
Python environment and package management

### [Jupyter](Jupyter.md)
Interactive computing environment

---

## References

- **Visual Studio Code Remote - SSH Documentation**: System requirements, setup, port forwarding, known limitations
- **GitHub Docs**: Generating SSH keys, adding to ssh-agent (macOS, Windows, Linux), key types and passphrases
- **Jupyter Overview**: Remote usage patterns, launching in conda environment, accessing via port forwarding
- **VS Code Source Control**: Git integration features (cloning/publishing, staging/committing, pushing/pulling, branches, PRs)
