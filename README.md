# NBA Win Predictor

A machine learning system that predicts NBA game outcomes with 82.6% accuracy using advanced feature engineering and multiple ML models.

## Features

- **82.6% Prediction Accuracy** using Logistic Regression
- **Real-time Predictions** with interactive Streamlit web app
- **13 Engineered Features** including rolling averages and opponent analysis
- **Multiple ML Models** (Logistic Regression, Random Forest, XGBoost)
- **Interactive Dashboard** with visualizations and analytics

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Archies/nba-win-predictor.git
cd nba-win-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Or use the deployment script:
```bash
./deploy_streamlit.bat  # Windows
```

## Usage

### Jupyter Notebook

1. Open `CSE_575.ipynb` in Jupyter Lab/Notebook
2. Run all cells sequentially to train the model
3. Use the prediction functions at the end

### Streamlit Web App

1. Launch the app using the installation steps above
2. Navigate to http://localhost:8501
3. Select two teams and predict game outcomes
4. View confidence scores and team analytics

### Python API

```python
from prediction_system import predict_game_simple

# Make a prediction
result = predict_game_simple('LAL', 'GSW', '2024-12-25', is_team_a_home=True)
print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Logistic Regression** | **82.6%** | **90.3%** | **82.8%** | **82.3%** | **82.5%** |
| Random Forest | 81.2% | 89.3% | 81.4% | 80.9% | 81.1% |
| XGBoost | 80.4% | 88.3% | 79.7% | 81.5% | 80.6% |

## Features

The model uses 13 engineered features:

- **Rolling Statistics** (5-game averages): FG%, 3P%, REB, AST, STL, BLK, TOV, PF, Wins
- **Opponent Analysis**: Recent opponent performance
- **Schedule Factors**: Rest days, back-to-back games
- **Home Court Advantage**: Home vs away performance

## Data

- **Source**: NBA API
- **Coverage**: 2023-24 and 2024-25 seasons
- **Records**: 4,910 team-game records
- **Games**: 2,455 unique games

## Project Structure

```
nba-win-predictor/
├── CSE_575.ipynb              # Main analysis notebook
├── streamlit_app.py           # Web application
├── nba_win_predictor.pkl      # Trained model
├── nba_team_games_combined.csv # Processed data
├── requirements.txt           # Dependencies
├── deploy_streamlit.bat       # Deployment script
├── README_DEPLOYMENT.md       # Deployment guide
└── Dockerfile                 # Docker configuration
```

## Development

### Training New Models

1. Run the Jupyter notebook `CSE_575.ipynb`
2. The notebook will:
   - Fetch latest NBA data
   - Engineer features
   - Train multiple models
   - Save the best model

### Adding Features

1. Modify the feature engineering section
2. Update the `features` list in the model training
3. Retrain and evaluate

### Deployment

#### Local Development
```bash
streamlit run streamlit_app.py
```

#### Docker Deployment
```bash
docker build -t nba-predictor .
docker run -p 8501:8501 nba-predictor
```

#### Production Deployment
See `README_DEPLOYMENT.md` for detailed deployment instructions.

## API Reference

### Core Functions

#### `predict_game_simple(team_a, team_b, game_date, is_team_a_home=True)`

Predicts the outcome of an NBA game.

**Parameters:**
- `team_a` (str): First team abbreviation (e.g., 'LAL')
- `team_b` (str): Second team abbreviation (e.g., 'GSW')
- `game_date` (str): Game date in 'YYYY-MM-DD' format
- `is_team_a_home` (bool): Whether team_a is playing at home

**Returns:**
```python
{
    'team_a': 'LAL',
    'team_b': 'GSW', 
    'predicted_winner': 'LAL',
    'confidence': 0.826,
    'team_a_win_prob': 0.826,
    'team_b_win_prob': 0.174,
    'team_a_recent_record': '4-1',
    'team_b_recent_record': '3-2'
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NBA API for providing comprehensive basketball data
- Scikit-learn for machine learning tools
- Streamlit for the web application framework
- XGBoost for gradient boosting implementation

## Contact

- **Author**: Archies
- **Email**: bhandaryarchies@gmail.com
- **Project Link**: https://github.com/Archies/nba-win-predictor

---

**Disclaimer**: This model is for educational and entertainment purposes only. Past performance does not guarantee future results.