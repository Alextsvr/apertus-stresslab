# Apertus StressLab - Windows environment probe (read-only).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\check_env.ps1
# Writes a plain-text report to results\env_check.txt (git-ignored).
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "results\env_check.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
& {
  "=== timestamp ==="; Get-Date -Format o
  "=== os ==="; (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | Format-List | Out-String).Trim()
  "=== cpu ==="; (Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String).Trim()
  "=== ram_gb ==="; [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  "=== gpu (WMI) ==="; (Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Format-List | Out-String).Trim()
  "=== nvidia-smi ==="
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv; nvidia-smi | Select-Object -First 5 } else { "nvidia-smi not found" }
  "=== disk C: free_gb ==="; [math]::Round((Get-PSDrive C).Free / 1GB, 1)
  "=== py launcher ==="
  if (Get-Command py -ErrorAction SilentlyContinue) { py -0p } else { "py launcher not found" }
  "=== python ==="
  if (Get-Command python -ErrorAction SilentlyContinue) { python --version; (Get-Command python).Source } else { "python not found" }
  "=== git ==="
  if (Get-Command git -ErrorAction SilentlyContinue) { git --version } else { "git not found" }
  "=== hf cache env ==="; "HF_HOME=$env:HF_HOME"; "HF_TOKEN set: " + [bool]$env:HF_TOKEN
} *>&1 | Out-File -FilePath $out -Encoding utf8
Write-Host "Wrote $out"
