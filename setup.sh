#!/bin/sh
# One-time setup after cloning the repo.
set -e

echo "== Configuring git hooks =="
git config core.hooksPath githooks
chmod +x githooks/post-merge

echo "== Setting up params.json =="
if [ ! -f params.json ] && [ -f params.default.json ]; then
    cp params.default.json params.json
    echo "Created params.json from params.default.json"
else
    echo "params.json already exists, leaving it untouched"
fi

echo ""
echo "Setup complete."
echo "params.json will auto-update with new parameters on every 'git pull'."