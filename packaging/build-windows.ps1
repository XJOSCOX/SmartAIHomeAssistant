param([switch]$SmokeTest)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildEnvironment = @{}
foreach ($name in @('UV_PROJECT_ENVIRONMENT', 'JAKE_BUILD_CONSOLE', 'YOLO_OFFLINE', 'YOLO_AUTOINSTALL', 'QT_QPA_PLATFORM')) {
    $buildEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $projectRoot
try {
    $env:UV_PROJECT_ENVIRONMENT = '.venv-build'
    $env:JAKE_BUILD_CONSOLE = '0'
    $env:YOLO_OFFLINE = 'true'
    $env:YOLO_AUTOINSTALL = 'false'
    uv sync --locked --extra desktop --extra detection --extra appearance --extra identity --extra packaging
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    uv run --no-sync pyinstaller --clean --noconfirm packaging/Jake.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
    if ($SmokeTest) {
        $env:QT_QPA_PLATFORM = 'offscreen'
        $smoke = Start-Process -FilePath (Join-Path $projectRoot 'dist/Jake/Jake.exe') -ArgumentList '--smoke-test' -WindowStyle Hidden -PassThru
        if (-not $smoke.WaitForExit(60000)) { $smoke.Kill(); throw 'Desktop smoke test timed out' }
        if ($smoke.ExitCode -ne 0) { throw "Desktop smoke failed: $($smoke.ExitCode)" }
        Write-Output 'Jake.exe offscreen startup/exit passed; no camera or store opened.'
    }
} finally {
    Pop-Location
    foreach ($name in $buildEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $buildEnvironment[$name], 'Process')
    }
}
