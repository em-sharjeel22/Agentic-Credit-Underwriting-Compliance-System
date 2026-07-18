$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$ingestScript = Join-Path $projectRoot "src\rag\ingest_documents.py"
$buildScript = Join-Path $projectRoot "src\rag\build_vectorstore.py"
$orchestratorScript = Join-Path $projectRoot "src\agents\orchestrator.py"

Push-Location $projectRoot
try {
    Write-Host "Running ingestion..."
    & $python $ingestScript

    Write-Host "Running vector store build..."
    & $python $buildScript

    Write-Host "Running underwriting workflow..."
    & $python $orchestratorScript
}
finally {
    Pop-Location
}
