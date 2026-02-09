$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[kb_chat] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[kb_chat] $msg" -ForegroundColor Yellow }

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Write-Info "Working dir: $here"

# Optional: auto-update if this folder is a git repo
if (Test-Path ".git") {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Info "Updating via git pull..."
    try {
      git pull --rebase
    } catch {
      Write-Warn "git pull failed (continuing). Error: $($_.Exception.Message)"
    }
  } else {
    Write-Warn "Found .git but git is not available in PATH; skip auto-update."
  }
} else {
  Write-Warn "Not a git repo (.git not found). Auto-update disabled (use git clone/pull if you want one-command updates)."
}

# Ensure venv
$venvPython = Join-Path $here ".venv\\Scripts\\python.exe"
if (!(Test-Path $venvPython)) {
  if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found in PATH. Install Python (or Anaconda) and retry."
  }
  Write-Info "Creating venv: .venv"
  python -m venv .venv
}

Write-Info "Installing requirements..."
& $venvPython -m pip install -U pip | Out-Host
& $venvPython -m pip install -r requirements.txt | Out-Host

# Streamlit server defaults:
# - 127.0.0.1: only this machine can access (safer)
# - 0.0.0.0: allow other devices in the same LAN to access
$addr = $env:KB_STREAMLIT_ADDR
if ([string]::IsNullOrWhiteSpace($addr)) { $addr = "127.0.0.1" }
$port = $env:KB_STREAMLIT_PORT
if ([string]::IsNullOrWhiteSpace($port)) { $port = "8501" }

Write-Info "Starting Streamlit on http://$addr`:$port"
& $venvPython -m streamlit run app.py --server.address $addr --server.port $port

