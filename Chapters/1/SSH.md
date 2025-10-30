# SSH: Secure Remote Access

## Overview

SSH (Secure Shell) provides encrypted remote login and command execution. It's the standard method for securely accessing remote servers and managing systems remotely.

---

## SSH Concepts

### Key Components
- **SSH Protocol**: Encrypted network protocol for secure communication
- **SSH Key Pair**: Consists of private key (kept secret) and public key (shared with servers)
- **Authentication Methods**: Key-based (recommended) vs password-based

### Why Key-Based Authentication?
- **Security**: More secure than passwords
- **Convenience**: No need to enter passwords repeatedly
- **Automation**: Enables scripted connections
- **Best Practice**: Industry standard for server access

---

## SSH Key Management

### 1. Generate SSH Keys

#### Create New Key Pair (Recommended: Ed25519)
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

#### Legacy System Fallback (RSA)
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

**Setup Instructions:**
- Accept default file path when prompted
- Set a strong passphrase (recommended for security)

### 2. Platform-Specific SSH Agent Setup

#### macOS
```bash
# Start ssh-agent
eval "$(ssh-agent -s)"

# Configure SSH config for keychain integration
# Add to ~/.ssh/config:
Host github.com
    AddKeysToAgent yes
    UseKeychain yes
    IdentityFile ~/.ssh/id_ed25519

# Add key to agent + keychain
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

#### Windows (OpenSSH in PowerShell)
```powershell
# Run elevated PowerShell (as admin)
Get-Service -Name ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent

# In normal terminal, add your key
ssh-add C:/Users/YOU/.ssh/id_ed25519
```

#### Linux
```bash
# Start ssh-agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519
```

---

## Key Distribution

### Add SSH Key to GitHub
Follow GitHub's "Adding a new SSH key to your GitHub account" guide after generating your key.

### Add SSH Key to Server
```bash
# Easiest method (if available)
ssh-copy-id username@remote_host

# Manual method
# 1. Copy your public key
cat ~/.ssh/id_ed25519.pub

# 2. Append to server's authorized_keys file
# SSH into server and run:
echo "your_public_key_content" >> ~/.ssh/authorized_keys

# 3. Set proper permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

---

## VS Code Remote - SSH

### Overview
VS Code Remote - SSH lets you open remote folders over SSH, installing a lightweight VS Code Server on the remote machine while using the full editor UI locally.

**Benefits:**
- Local-quality development experience on remote compute
- Full IntelliSense, navigation, and debugging capabilities
- Supported on Linux, macOS, and Windows

### Prerequisites
- Compatible OpenSSH client locally
- SSH server on remote host
- VS Code and Remote-SSH extension

### Setup Steps

#### 1. Test Plain SSH Connection
```bash
ssh username@remote_host
# Accept host fingerprint on first connect
```

#### 2. Install Required Software
- Install VS Code
- Install Remote-SSH extension (or Remote Development extension pack)

#### 3. Configure Remote Connection

**Method 1: VS Code Interface**
1. Run "Remote-SSH: Add New SSH Host…" from Command Palette
2. Enter `username@hostname` or full ssh command
3. Choose config file location

**Method 2: Manual SSH Config**
Edit `~/.ssh/config`:
```
Host myserver
    HostName server.example.com
    User username
    IdentityFile ~/.ssh/id_ed25519
```

#### 4. Connect to Remote Host
1. Run "Remote-SSH: Connect to Host…"
2. Select your configured host
3. VS Code installs server components on first connect

---

## Configuration Tips

### SSH Config File (`~/.ssh/config`)
```bash
# Example configuration
Host myserver
    HostName 192.168.1.100
    User myuser
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 8888 localhost:8888  # For Jupyter port forwarding
```

### Security Best Practices
- Use key-based authentication (disable password auth on server)
- Set proper file permissions:
  - Private key: `chmod 600 ~/.ssh/id_ed25519`
  - SSH directory: `chmod 700 ~/.ssh`
- Use strong passphrases on private keys
- Regularly rotate SSH keys

### VS Code Remote Settings
Configure default extensions to auto-install on remote hosts:
```json
"remote.SSH.defaultExtensions": [
    "ms-python.python",
    "ms-toolsai.jupyter"
]
```

---

## Troubleshooting

### Common Issues

#### Connection Refused
- Check SSH service is running on remote host
- Verify firewall settings
- Confirm correct port and hostname

#### Permission Denied
- Verify SSH key is added to ssh-agent
- Check key permissions (should be 600)
- Ensure public key is in remote `authorized_keys`

#### "Bad Permissions" Errors
- Fix SSH directory permissions:
  ```bash
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/*
  ```

#### Port Forwarding Issues
- Use VS Code's Ports view for easy management
- Configure `LocalForward` in SSH config for persistent forwarding
- Check local port availability

---

## Integration with Development Workflow

### Typical Use Cases:
1. **Remote Development**: Code on powerful remote servers
2. **Jupyter Access**: Run notebooks on remote compute
3. **File Transfer**: Secure copy files with `scp` or `rsync`
4. **Tunneling**: Access remote services securely

### Related Tools:
- **SCP**: Secure file copy
- **RSYNC**: Efficient file synchronization
- **SSHFS**: Mount remote filesystems locally
- **Port Forwarding**: Access remote services
