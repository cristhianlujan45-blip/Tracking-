# Forja3D: instalación en Windows con GPU NVIDIA (pensado para RTX serie 50, p. ej. RTX 5060 8 GB)
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\instalar.ps1            # forma 3D + conceptos
#   powershell -ExecutionPolicy Bypass -File scripts\windows\instalar.ps1 -Texture   # + texturas (experimental)
#
# Requisitos previos: Python 3.11 (python.org, marca "Add to PATH"), Git, y el driver NVIDIA actualizado.
# Para -Texture además: Visual Studio 2022 Build Tools ("Desarrollo de escritorio con C++") y CUDA Toolkit 12.8.

param(
    [switch]$Texture,
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Venv = Join-Path $Root ".venv"
$VPy = Join-Path $Venv "Scripts\python.exe"
$Vendor = Join-Path $Root "gpu_worker\vendor"
$Hunyuan = Join-Path $Vendor "Hunyuan3D-2"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { Fail "Falló: $exe $($argv -join ' ')" }
}

Step "Comprobando herramientas"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "Git no está instalado (https://git-scm.com)" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "No se encuentra nvidia-smi: instala/actualiza el driver de NVIDIA" }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

Step "Creando entorno virtual en .venv"
if (-not (Test-Path $VPy)) {
    $pyParts = @($Python.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries))
    $pyArgs = @()
    if ($pyParts.Length -gt 1) { $pyArgs = $pyParts[1..($pyParts.Length - 1)] }
    & $pyParts[0] @($pyArgs + @("-m", "venv", $Venv))
    if ($LASTEXITCODE -ne 0) { Fail "No se pudo crear el entorno con '$Python'. Instala Python 3.11 o usa -Python python" }
}
Run $VPy @("-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools")

Step "Instalando PyTorch con CUDA 12.8 (necesario para las RTX serie 50 / Blackwell)"
Run $VPy @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")

Step "Descargando Hunyuan3D-2 (Tencent)"
New-Item -ItemType Directory -Force -Path $Vendor | Out-Null
if (-not (Test-Path $Hunyuan)) {
    Run git @("clone", "--depth", "1", "https://github.com/Tencent/Hunyuan3D-2", $Hunyuan)
}
Run $VPy @("-m", "pip", "install", "-r", (Join-Path $Hunyuan "requirements.txt"))
Run $VPy @("-m", "pip", "install", "-e", $Hunyuan)

Step "Instalando la app y el worker"
Run $VPy @("-m", "pip", "install", "-r", (Join-Path $Root "backend\requirements.txt"))
Run $VPy @("-m", "pip", "install", "-r", (Join-Path $Root "gpu_worker\requirements.txt"))

Step "Verificando que PyTorch ve tu GPU"
# Si otro paquete reemplazó torch por una versión sin CUDA, se reinstala la buena.
& $VPy -c "import torch,sys; ok=torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list(); sys.exit(0 if ok else 3)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Reinstalando PyTorch cu128…" -ForegroundColor Yellow
    Run $VPy @("-m", "pip", "install", "--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")
}
Run $VPy @("-c", "import torch; p=torch.cuda.get_device_properties(0); print(f'GPU: {p.name} · {p.total_memory/1024**3:.1f} GB · CUDA {torch.version.cuda} · torch {torch.__version__}')")

if ($Texture) {
    Step "Compilando extensiones de textura (experimental)"
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { Fail "Falta nvcc: instala CUDA Toolkit 12.8 y abre una terminal nueva" }
    Push-Location (Join-Path $Hunyuan "hy3dgen\texgen\custom_rasterizer")
    Run $VPy @("setup.py", "install")
    Pop-Location
    Push-Location (Join-Path $Hunyuan "hy3dgen\texgen\differentiable_renderer")
    Run $VPy @("setup.py", "install")
    Pop-Location
}

Step "Configurando .env"
$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Root ".env.example") $EnvFile
}
(Get-Content $EnvFile) -replace '^GEN_PROVIDER=.*', 'GEN_PROVIDER=local' | Set-Content $EnvFile

Write-Host "`nListo. Inicia todo con:  scripts\windows\iniciar.bat" -ForegroundColor Green
Write-Host "La primera generación descarga los modelos (varios GB) desde Hugging Face; las siguientes son rápidas."
