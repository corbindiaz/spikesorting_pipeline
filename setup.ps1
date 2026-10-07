Write-Host "== Configuring git hooks =="
git config core.hooksPath githooks

Write-Host "== Setting up params.json =="
if (!(Test-Path "params.json") -and (Test-Path "params.default.json")) {
    Copy-Item "params.default.json" "params.json"
    Write-Host "Created params.json from params.default.json"
}
else {
    Write-Host "params.json already exists, leaving it untouched"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "params.json will auto-update with new parameters on every 'git pull'."