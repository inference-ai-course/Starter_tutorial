# Git: Local + VS Code Integration

## Overview

Git is a distributed version control system that tracks changes in your codebase over time. It enables collaboration, version history, and safe experimentation with code changes.

---

## Why Git for Version Control?

**Key Benefits:**
- **Change Tracking**: Maintain complete history of modifications
- **Collaboration**: Work with others without overwriting each other's work
- **Version Recovery**: Revert to previous versions if something breaks
- **Branching**: Experiment with new features in isolated branches
- **Backup**: Distributed nature provides multiple copies of your code

---

## What is GitHub?

GitHub is a cloud-based hosting platform for Git repositories that provides:
- Central location to store code online
- Collaboration features (pull requests, code review)
- Issue tracking and project management
- CI/CD integration
- Community features

**Git vs GitHub:** Git manages version control locally; GitHub hosts repositories online and adds collaboration features.

---

## Common Git Commands

### Basic Operations
| Command | Description |
|---------|-------------|
| `git init` | Initialize a new Git repository |
| `git add` | Stage changes for the next commit |
| `git commit` | Save staged changes with a descriptive message |
| `git status` | Check modified or staged files |
| `git branch` | Create, list, or delete branches |
| `git checkout` | Switch between branches or restore files |
| `git push` | Upload local commits to remote repository |
| `git pull` | Download and merge changes from remote |
| `git merge` | Combine changes from different branches |

---

## Using Git with VS Code

VS Code has built-in Source Control integration that makes Git operations visual and intuitive.

### 1. Sign in to GitHub in VS Code

#### Setup Steps:
1. **Open Accounts Menu**: In VS Code, open the Accounts menu in the Activity Bar and select "Sign in with GitHub"
2. **Copilot Setup**: If using Copilot, hover the Copilot icon in Status Bar → "Set up Copilot" → Follow GitHub sign-in prompts
3. **Install Extensions**: Install "GitHub Pull Requests and Issues" extension → Authenticate in browser → Return to VS Code
4. **Token Authentication**: If not auto-redirected, copy authorization token → Go to VS Code → Select "Signing in to github.com..." → Paste token → Press Enter
5. **Clone Repository**: Use "Git: Clone" from Command Palette → Choose "Clone from GitHub" → Sign in → Select repository

#### Benefits of Signing In:
- Create PRs from Pull Requests view
- Add reviewers/labels
- Review diffs and comment
- Merge directly from VS Code
- Browse issues and create branches from issues
- Copilot can generate PR titles/descriptions

---

### 2. Troubleshooting Sign-in Issues

#### Common Solutions:
- **Browser Issues**: Set a working default browser; switching browsers may resolve infinite "Signing in to github.com..." loops
- **Corporate Proxy**: Configure VS Code's proxy in settings.json:
  ```json
  "http.proxy": "http://username:password@proxyhost:port"
  ```
- **Linux Firefox**: Ensure browser binary is in PATH (symlink Flatpak binary if needed)
- **Credential Prompts**: Configure Git credential helper (e.g., Windows Credential Manager)

---

### 3. Quick Checklist

#### Essential Setup:
1. Install Git and VS Code
2. Install "GitHub Pull Requests and Issues" extension
3. Sign in via Accounts menu or when prompted during GitHub actions
4. Start cloning, committing, pushing, creating PRs, and managing issues

#### Pro Tips:
- Use key-based authentication for better security
- Make small, frequent commits with clear messages
- Use branches for new features
- Review changes before committing
- Write meaningful commit messages
