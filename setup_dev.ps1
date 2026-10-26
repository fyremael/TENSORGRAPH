# setup_dev.ps1
# AntiGravity: TENSORGRAPH Bootstrapping Script (v1.0.0)
$GCT_BANNER = @"
--------------------------------------------------
   GCT_SYSTEM: TENSORGRAPH
   STATUS: INITIALIZING (Strict uv Workflow)
--------------------------------------------------
"@
Write-Host $GCT_BANNER -ForegroundColor Cyan

# 1. Path Verification
$UV_PATH = "$env:USERPROFILE\.local\bin"
if ($env:Path -notlike "*$UV_PATH*") {
    $env:Path = "$UV_PATH;$env:Path"
}

# 2. Virtual Environment Cleanup & Creation
if (Test-Path ".venv-aether") { Remove-Item -Recurse -Force .venv-aether }
if (!(Test-Path ".venv")) {
    Write-Host "[STEP] Creating isolated venv..." -ForegroundColor Gray
    uv venv --python 3.12
}

# 3. Dependency Sync
Write-Host "[STEP] Synchronizing dependencies..." -ForegroundColor Gray
uv sync --all-extras

# 4. Final Verification
$pythonVer = & .venv\Scripts\python --version
Write-Host "[OK] Workspace Active: TENSORGRAPH" -ForegroundColor Green
Write-Host "[OK] Interpreter: $pythonVer" -ForegroundColor Green

Write-Host "`nReady for Frontier Engineering." -ForegroundColor Cyan
