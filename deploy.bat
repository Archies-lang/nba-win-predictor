@echo off
echo NBA Predictor Production Deployment
echo ======================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker first.
    exit /b 1
)

echo Docker environment verified

REM Build and deploy
echo Building NBA Predictor image...
docker build -t nba-predictor:latest .

if %errorlevel% equ 0 (
    echo Image built successfully
) else (
    echo Image build failed
    exit /b 1
)

echo Starting production deployment...
docker-compose up -d

if %errorlevel% equ 0 (
    echo Production deployment successful!
    echo.
    echo Access Information:
    echo   * Streamlit Demo: http://localhost:8501
    echo   * Flask API: http://localhost:8000
    echo   * Redis: localhost:6379
    echo.
    echo Management Commands:
    echo   * View logs: docker-compose logs -f nba-predictor
    echo   * Stop services: docker-compose down
    echo   * Restart: docker-compose restart
) else (
    echo Deployment failed
    exit /b 1
)