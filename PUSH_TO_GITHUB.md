# Instructions to Push to GitHub

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `dry-eye-disease-detection` (or your preferred name)
3. Choose **Public** or **Private**
4. **DO NOT** check "Initialize this repository with a README" (we already have files)
5. Click **"Create repository"**

## Step 2: Copy Your Repository URL

After creating the repository, GitHub will show you a URL like:
- `https://github.com/sweetyash/dry-eye-disease-detection.git` (HTTPS)
- `git@github.com:sweetyash/dry-eye-disease-detection.git` (SSH)

## Step 3: Run These Commands

Replace `YOUR_REPO_URL` with your actual repository URL:

```bash
# Add the remote repository
git remote add origin YOUR_REPO_URL

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: If you already have a repository

If you already created the repository, just run:

```bash
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```

## Notes

- If you're using HTTPS, GitHub may ask for your username and a Personal Access Token (not password)
- If you're using SSH, make sure your SSH key is set up with GitHub
- The `.gitignore` file has been configured to exclude:
  - Database files (`.db`, `.sqlite`)
  - Model files (`.h5`, `.pkl`)
  - Python cache files (`__pycache__/`)
  - Virtual environments
  - OS-specific files

