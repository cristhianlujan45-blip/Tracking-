# Forja3D: inicia el worker de IA (GPU) y la app, y abre el navegador.
#
#   scripts\windows\iniciar.bat                 # forma 3D + conceptos en tu GPU
#   scripts\windows\iniciar.bat -Texture        # + texturas (requiere instalar con -Texture; lento con 8 GB)
#   scripts\windows\iniciar.bat -Multiview      # + varias vistas con Hunyuan3D-2mv
#   scripts\windows\iniciar.bat -Demo           # sin IA: comprueba que todo funciona

param(
    [switch]$Texture,
    [switch]$Multiview,
    [switch]$Demo,
    [string]$ImageModel = "stabilityai/sd-turbo",
    [int]$Port = 8000,
    [int]$WorkerPort = 8001
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { Write-Host "Primero ejecuta scripts\windows\instalar.bat" -ForegroundColor Red; exit 1 }

# Evita fragmentar la VRAM en tarjetas de 8 GB
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$workerArgs = @("-m", "worker", "--port", "$WorkerPort", "--image-model", $ImageModel)
if ($Demo) { $workerArgs += @("--backend", "dummy", "--texture", "--multiview") }
if ($Texture) { $workerArgs += "--texture" }
if ($Multiview) { $workerArgs += "--multiview" }

Write-Host "Iniciando worker de IA en http://127.0.0.1:$WorkerPort …" -ForegroundColor Cyan
$worker = Start-Process -FilePath $VPy -ArgumentList $workerArgs -WorkingDirectory (Join-Path $Root "gpu_worker") -PassThru

# Espera a que el worker responda (la primera vez importa torch y tarda un poco)
$deadline = (Get-Date).AddSeconds(120)
do {
    Start-Sleep -Seconds 2
    try { $health = Invoke-RestMethod "http://127.0.0.1:$WorkerPort/health" -TimeoutSec 2 } catch { $health = $null }
} until ($health -or (Get-Date) -gt $deadline -or $worker.HasExited)
if (-not $health) { Write-Host "El worker no arrancó; revisa su ventana para ver el error." -ForegroundColor Red; exit 1 }
Write-Host "Worker listo: $($health.gpu) · $($health.capabilities -join ', ')" -ForegroundColor Green

$env:GEN_PROVIDER = "local"
$env:LOCAL_WORKER_URL = "http://127.0.0.1:$WorkerPort"
try { Start-Process "http://localhost:$Port" } catch { }  # abrir el navegador es opcional
Write-Host "App en http://localhost:$Port  (Ctrl+C para salir)" -ForegroundColor Green
try {
    Push-Location (Join-Path $Root "backend")
    & $VPy -m uvicorn app.main:app --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
    if (-not $worker.HasExited) { Stop-Process -Id $worker.Id }
}
