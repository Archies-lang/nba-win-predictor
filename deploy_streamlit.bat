@echo off
echo 🏀 NBA Win Predictor - Deployment Script
echo ========================================

echo Installing required packages...
pip install -r requirements.txt --quiet

echo Checking if model exists...
if not exist "nba_win_predictor.pkl" (
    echo ❌ Model file not found! Please train the model first by running the notebook.
    pause
    exit /b 1
)

echo Checking if data exists...
if not exist "nba_team_games_combined.csv" (
    echo ❌ Data file not found! Please run data collection in the notebook first.
    pause
    exit /b 1
)

echo ✅ All dependencies ready!
echo 🚀 Launching NBA Win Predictor Streamlit App...
echo.
echo The app will be available at:
echo   Local URL: http://localhost:8501
echo   Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py --server.headless true

pause