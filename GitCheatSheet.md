# 🧩 Git Cheat Sheet

A quick reference for Git commands for version control and collaboration.

---

## 📂 Repository Setup

| Task | Command |
|------|---------|
| Initialize a new Git repo | `git init` |
| Clone an existing repo | `git clone <repo_url>` |
| Check remote URLs | `git remote -v` |

---

## 🌱 Branching

| Task | Command |
|------|---------|
| List branches | `git branch` |
| Switch to branch | `git switch <branch>` or `git checkout <branch>` |
| Create a new branch | `git branch <new_branch>` |
| Create & switch to branch | `git checkout -b <new_branch>` |
| Delete a branch | `git branch -d <branch>` |
| Fetch all branches from remote | `git fetch origin` |

---

## ✏️ Making Changes

| Task | Command |
|------|---------|
| Check status | `git status` |
| Stage a file | `git add <file>` |
| Stage all changes | `git add .` |
| Commit staged changes | `git commit -m "Commit message"` |
| Amend last commit | `git commit --amend` |
| View commit history | `git log` or `git log --oneline --graph --all` |

---

## 🔄 Sync with Remote

| Task | Command |
|------|---------|
| Push branch to remote | `git push origin <branch>` |
| Pull latest changes | `git pull origin <branch>` |
| Set upstream for a branch | `git push --set-upstream origin <branch>` |

---

## 🔧 Undo & Revert

| Task | Command |
|------|---------|
| Unstage a file | `git restore --staged <file>` |
| Discard local changes | `git restore <file>` |
| Revert a commit | `git revert <commit_hash>` |
| Reset to previous commit | `git reset --hard <commit_hash>` |

---

## 🔍 Inspecting

| Task | Command |
|------|---------|
| Show changes | `git diff` |
| Show staged changes | `git diff --staged` |
| Show remote branches | `git branch -r` |
| Show commit details | `git show <commit_hash>` |

---

## 🤝 Collaboration

| Task | Command |
|------|---------|
| Fetch latest from remote | `git fetch` |
| Merge branch | `git merge <branch>` |
| Rebase branch | `git rebase <branch>` |
| Resolve merge conflicts | Edit files → `git add <file>` → `git commit` |

---

## 💡 Tips

- Use `git log --oneline --graph --all --decorate` for a visual branch history  
- Use `git status` often to avoid surprises  
- Comment commits clearly for teamwork  

---

*Saved as `git_cheatsheet.md` for quick reference.*
