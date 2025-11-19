# 🏀 NBA Win Predictor - Streamlit Deployment

## 🚀 Quick Start

### Option 1: Using Deployment Script (Recommended)
```bash
# Simply double-click deploy_streamlit.bat or run:
deploy_streamlit.bat
```

### Option 2: Manual Deployment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run streamlit_app.py
```

## 📋 Prerequisites

Make sure you have these files in your project directory:
- ✅ `nba_win_predictor.pkl` (trained model)
- ✅ `nba_team_games_combined.csv` (NBA game data)
- ✅ `streamlit_app.py` (Streamlit application)
- ✅ `requirements.txt` (Python dependencies)

## 🌐 Accessing the App

Once deployed, the app will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

## 🎯 Features

### 📊 Live Predictions
- Select any two NBA teams
- Choose game date and home court advantage
- Get real-time predictions with confidence scores
- View team recent form and win probabilities

### 📈 System Dashboard  
- Model accuracy tracking over time
- API response time monitoring
- System performance metrics

### 🔍 Model Analytics
- Feature importance analysis
- Model comparison metrics
- Performance breakdowns

### ⚡ Performance Monitor
- Load testing capabilities
- System resource monitoring
- Real-time performance metrics

### 💼 Business Intelligence
- Revenue and user analytics
- Growth tracking
- Feature usage statistics

## 🔧 Model Performance

- **Accuracy**: 82.6%
- **AUC Score**: 90.3%
- **Features**: 13 engineered features
- **Models**: Logistic Regression (best), Random Forest, XGBoost

## 🛠️ Troubleshooting

### Model Not Found Error
```
❌ Model file 'nba_win_predictor.pkl' not found
```
**Solution**: Run the Jupyter notebook `CSE_575.ipynb` to train and save the model.

### Data Not Found Error  
```
❌ Data file 'nba_team_games_combined.csv' not found
```
**Solution**: Run the data collection cells in the notebook to download NBA data.

### Port Already in Use
```
Address already in use
```
**Solution**: Either stop the existing Streamlit app or use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 📁 File Structure
```
CSE_575/
├── streamlit_app.py           # Main Streamlit application
├── nba_win_predictor.pkl      # Trained ML model
├── nba_team_games_combined.csv # NBA game data
├── requirements.txt           # Python dependencies
├── deploy_streamlit.bat       # Deployment script
├── CSE_575.ipynb            # Model training notebook
└── README_DEPLOYMENT.md       # This file
```

## 🔒 Security Notes

- The app runs locally by default (localhost:8501)
- For production deployment, configure proper authentication
- Use HTTPS in production environments
- Consider rate limiting for public deployments

## 🎉 Success!

Your NBA Win Predictor is now deployed and ready to make accurate game predictions!

Visit http://localhost:8501 to start predicting NBA games! 🏀