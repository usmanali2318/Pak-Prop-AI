# 🏠 Pakistan AI House Price Prediction & Market Analytics

A Streamlit web application for predicting house prices and exploring real estate market trends across major Pakistani cities using machine learning.

## Features

- **Price Prediction** — Estimate property prices based on city, location, area, and property type
- **Market Analytics** — Interactive charts and statistics across cities and property types
- **Model Insights** — Feature importance, model performance metrics, and cross-validation scores

## Tech Stack

- **Frontend:** Streamlit
- **ML Models:** XGBoost, scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualizations:** Plotly
- **Fuzzy Matching:** RapidFuzz (for location normalization)

## Project Structure

```
pak-prop-ai/
├── app.py                  # Main Streamlit application (all pages)
├── model_training.py       # Model training script — run once before the app
├── utils.py                # Shared helpers, constants, unit conversions
├── requirements.txt        # Python dependencies
├── data/                   # Place your dataset CSV(s) here (see below)
├── models/                 # Trained model saved here after training
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/pak-prop-ai.git
cd pak-prop-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place one of the following CSV files inside the `data/` folder (the training script tries each in order):

| File | Description |
|------|-------------|
| `zameen-updated.csv` | Preferred — cleaned Zameen.com listings |
| `Property_with_Feature_Engineering.csv` | Pre-engineered features |
| `raw_data_zameen.csv` | Raw Zameen scrape |
| `property_data.csv` | Generic property data |
| `all_listings_clean.csv` | Cleaned all-listings export |
| `zameen.csv` | Fallback raw Zameen CSV |

Datasets are not included in this repository due to file size. You can source them from [Kaggle](https://www.kaggle.com/search?q=zameen+pakistan+property) or your own Zameen.com scrape.

### 4. Train the model

```bash
python model_training.py
```

This saves `models/best_model.pkl` which the app loads at startup.

### 5. Run the app

```bash
streamlit run app.py
```

## Data Notes

- Prices are in **PKR (Pakistani Rupees)**
- Area units supported: **Marla, Kanal, Square Feet, Square Yards**
- Cities: Karachi, Lahore, Islamabad, Rawalpindi, Peshawar, Faisalabad, and more

## License

MIT
