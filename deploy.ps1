# Deploy Script (Debug)
Write-Host "Starting Debug Deployment..."

# 1. Start Backend with Logging
Write-Host "Starting Backend..."
$backendProcess = Start-Process -FilePath "python" -ArgumentList "quantum_swarmvla_backend.py" -WorkingDirectory "backend" -PassThru -WindowStyle Hidden -RedirectStandardOutput "..\backend_out.log" -RedirectStandardError "..\backend_err.log"

# 2. Start Frontend
Write-Host "Starting Frontend..."
$frontendProcess = Start-Process -FilePath "cmd" -ArgumentList "/c npx serve -s build -l 3000" -WorkingDirectory "frontend" -PassThru -WindowStyle Hidden -RedirectStandardOutput "..\frontend_out.log" -RedirectStandardError "..\frontend_err.log"

# Wait for servers
Write-Host "Waiting 20s for servers..."
Start-Sleep -Seconds 20

# 3. Run DOM Agent
Write-Host "Running DOM Agent Verification..."
$env:DEPLOY_URL = "http://localhost:3000"
$env:REACT_APP_API_URL = "http://127.0.0.1:5000"

python qa_agent/dom_agent.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment Verified Successfully!" -ForegroundColor Green
}
else {
    Write-Host "Deployment Failed Verification!" -ForegroundColor Red
}

# Cleanup
Write-Host "Stopping Servers..."
Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
