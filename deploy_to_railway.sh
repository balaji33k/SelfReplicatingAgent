#!/bin/bash
# deploy_to_railway.sh — Push to GitHub and deploy to Railway
# Run this from a terminal that has git + gh (GitHub CLI) installed.
# Usage: bash deploy_to_railway.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="SelfReplicatingAgent"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Self-Replicating Agent — Railway Deploy    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Step 1: Git init
if [ ! -d ".git" ]; then
  echo "▶ Initialising git repo..."
  git init
  git config user.email "balaji33.k@gmail.com"
  git config user.name "Balaji Kasula"
else
  echo "✓ Git repo already initialised"
fi

# Step 2: Commit
echo "▶ Staging files..."
git add .

if git diff --cached --quiet; then
  echo "✓ Nothing new to commit"
else
  git commit -m "Self-replicating AI agent — multi-gen evolution with live dashboard

  - 7-agent specialisation pipeline (Analyst→Coder→Critic→Debugger)
  - Groq/Llama-3.3 backend — zero Gemini dependency
  - Live MISSION_CONTROL dashboard (polls data/ every 3s)
  - Clone Inspector agent — semantic originality check on each spawn
  - Per-file evolution engine — stays within Groq free-tier TPM budget

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  echo "✓ Committed"
fi

# Step 3: Create GitHub repo (requires gh CLI: https://cli.github.com)
if command -v gh &>/dev/null; then
  echo "▶ Creating GitHub repo '$REPO_NAME'..."
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push \
    --description "Self-replicating AI coding agent with live evolution dashboard" \
    2>/dev/null || true
  git push -u origin main 2>/dev/null || git push -u origin master
  GITHUB_URL=$(gh repo view --json url -q .url)
  echo "✓ Pushed to GitHub: $GITHUB_URL"
else
  echo ""
  echo "gh CLI not found. Create a GitHub repo manually:"
  echo "  1. Go to https://github.com/new"
  echo "  2. Name it: $REPO_NAME"
  echo "  3. Run:"
  echo "       git remote add origin https://github.com/<your-user>/$REPO_NAME.git"
  echo "       git branch -M main"
  echo "       git push -u origin main"
  echo ""
  read -p "Press Enter once you've pushed to GitHub to continue with Railway..."
  GITHUB_URL="https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]//' | sed 's/\.git//')"
fi

# Step 4: Railway deploy instructions
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         Deploy to Railway (2 minutes)        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "1. Go to https://railway.app  → New Project → Deploy from GitHub repo"
echo "   Select: $REPO_NAME"
echo ""
echo "2. Add environment variable:"
echo "   Key:   GROQ_API_KEY"
echo "   Value: (your Groq API key from console.groq.com)"
echo ""
echo "3. Railway will auto-detect Procfile and deploy."
echo "   Your dashboard URL will be:"
echo "   https://<your-project>.up.railway.app/MISSION_CONTROL.html"
echo ""
echo "4. Share that URL with your team — the dashboard auto-refreshes every 3s."
echo ""
echo "Done! ✓"
