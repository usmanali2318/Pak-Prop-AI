"""
Pakistan House Price Prediction - Model Training Script
Run this script once to train the model before launching the app.
Usage: python model_training.py

PAYLOAD CONTRACT (keys saved to best_model.pkl):
  model                   — trained best model object
  model_name              — name of best model (str)
  label_encoder_city      — LabelEncoder for city
  label_encoder_ptype     — LabelEncoder for property_type
  location_median_price_map — dict: location -> median price (PKR)
  global_median_price     — float: fallback when location unknown
  city_ptype_median_map   — dict: (city, ptype) -> median price (PKR)
  feature_cols            — list of feature column names used
  feature_names           — human-readable feature names
  feature_importances     — array or None
  results                 — dict: model_name -> {R2, MAE, RMSE}
  cv_scores               — array of 5-fold CV R² scores for best model
  y_test_log              — actual log prices (test set)
  y_pred_log              — predicted log prices (test set)
  y_test_pkr              — actual prices in PKR (test set)
  y_pred_pkr              — predicted prices in PKR (test set)
  residuals               — y_test_log - y_pred_log
  df_clean                — DataFrame with cleaned market data
  city_stats              — city × property_type aggregates
  area_cat_stats          — city × ptype × area_category aggregates
  area_cat_labels         — dict: int -> label string
  total_records           — int
  cities_supported        — sorted list of city strings
  property_types          — sorted list of property type strings
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_model.pkl"

sys.path.insert(0, str(Path(__file__).parent))
from utils import convert_to_marla, parse_area_string, normalize_location, MARLA_TO_SQFT

# ─── 1. LOAD & MERGE ALL DATASETS ───────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading datasets...")
print("=" * 60)

all_frames = []

def load_zameen_updated():
    path = DATA_DIR / "zameen-updated.csv"
    if not path.exists():
        return None
    print("  Loading zameen-updated.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'Area Type': 'area_type', 'Area Size': 'area_size', 'Area Category': 'area_category'})
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']    = pd.to_numeric(df['baths'], errors='coerce')
    df['price']    = pd.to_numeric(df['price'], errors='coerce')
    parsed = df['area'].apply(parse_area_string)
    df['_area_val']  = parsed.apply(lambda x: x[0])
    df['_area_unit'] = parsed.apply(lambda x: x[1])
    df['area_marla'] = df.apply(lambda r: convert_to_marla(r['_area_val'], r['_area_unit']), axis=1)
    df['area_sqft']  = df['area_marla'] * MARLA_TO_SQFT
    df['area_sqyd']  = df['area_sqft'] / 9.0
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

def load_property_fe():
    path = DATA_DIR / "Property_with_Feature_Engineering.csv"
    if not path.exists():
        return None
    print("  Loading Property_with_Feature_Engineering.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df['bedrooms']   = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']      = pd.to_numeric(df['baths'], errors='coerce')
    df['price']      = pd.to_numeric(df['price'], errors='coerce')
    df['area_marla'] = pd.to_numeric(df['area_marla'], errors='coerce')
    df['area_sqft']  = pd.to_numeric(df['area_sqft'], errors='coerce')
    df['area_sqyd']  = df['area_sqft'] / 9.0
    if 'location_city' in df.columns:
        df = df.rename(columns={'location_city': 'city'})
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

def load_raw_zameen():
    path = DATA_DIR / "raw_data_zameen.csv"
    if not path.exists():
        return None
    print("  Loading raw_data_zameen.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'type': 'property_type', 'bedroom': 'bedrooms',
                             'bath': 'baths', 'location_city': 'city', 'added': 'date_added'})
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']    = pd.to_numeric(df['baths'], errors='coerce')
    df['price']    = pd.to_numeric(df['price'], errors='coerce')
    parsed = df['area'].apply(parse_area_string)
    df['_area_val']  = parsed.apply(lambda x: x[0])
    df['_area_unit'] = parsed.apply(lambda x: x[1])
    df['area_marla'] = df.apply(lambda r: convert_to_marla(r['_area_val'], r['_area_unit']), axis=1)
    df['area_sqft']  = df['area_marla'] * MARLA_TO_SQFT
    df['area_sqyd']  = df['area_sqft'] / 9.0
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

def load_property_data():
    path = DATA_DIR / "property_data.csv"
    if not path.exists():
        return None
    print("  Loading property_data.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'type': 'property_type', 'bedroom': 'bedrooms',
                             'bath': 'baths', 'location_city': 'city', 'added': 'date_added'})
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']    = pd.to_numeric(df['baths'], errors='coerce')
    df['price']    = pd.to_numeric(df['price'], errors='coerce')
    parsed = df['area'].apply(parse_area_string)
    df['_area_val']  = parsed.apply(lambda x: x[0])
    df['_area_unit'] = parsed.apply(lambda x: x[1])
    df['area_marla'] = df.apply(lambda r: convert_to_marla(r['_area_val'], r['_area_unit']), axis=1)
    df['area_sqft']  = df['area_marla'] * MARLA_TO_SQFT
    df['area_sqyd']  = df['area_sqft'] / 9.0
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

def load_all_listings():
    path = DATA_DIR / "all_listings_clean.csv"
    if not path.exists():
        return None
    print("  Loading all_listings_clean.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'beds': 'bedrooms', 'size_marla': 'area_marla', 'price_pkr': 'price'})
    df['bedrooms']   = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']      = pd.to_numeric(df['baths'], errors='coerce')
    df['price']      = pd.to_numeric(df['price'], errors='coerce')
    df['area_marla'] = pd.to_numeric(df['area_marla'], errors='coerce')
    df['area_sqft']  = df['area_marla'] * MARLA_TO_SQFT
    df['area_sqyd']  = df['area_sqft'] / 9.0
    df['property_type'] = 'House'
    df['purpose']    = 'For Sale'
    df['date_added'] = None
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

def load_zameen_csv():
    path = DATA_DIR / "zameen.csv"
    if not path.exists():
        return None
    print("  Loading zameen.csv ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'type': 'property_type'})
    df['bedrooms']  = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['baths']     = pd.to_numeric(df['baths'], errors='coerce')
    df['price']     = pd.to_numeric(df['price'], errors='coerce')
    df['area_sqft'] = pd.to_numeric(df['area_sqft'], errors='coerce')
    df['area_marla'] = df['area_sqft'] / MARLA_TO_SQFT
    df['area_sqyd']  = df['area_sqft'] / 9.0
    df['location']   = df.get('address', '')
    df['purpose']    = 'For Sale'
    if 'date_added' not in df.columns:
        df['date_added'] = None
    return df[['city','location','property_type','bedrooms','baths','price',
               'area_marla','area_sqft','area_sqyd','purpose','date_added']].copy()

for loader in [load_zameen_updated, load_property_fe, load_raw_zameen,
               load_property_data, load_all_listings, load_zameen_csv]:
    try:
        frame = loader()
        if frame is not None and len(frame) > 0:
            all_frames.append(frame)
            print(f"    Loaded {len(frame):,} rows")
    except Exception as e:
        print(f"    Error in loader: {e}")

if not all_frames:
    print("ERROR: No dataset files found in data/ folder.")
    print("Please place your Zameen CSV files in the data/ directory.")
    sys.exit(1)

df_raw = pd.concat(all_frames, ignore_index=True)
print(f"\n  Combined total: {len(df_raw):,} rows")

# ─── 2. CLEANING ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Cleaning data...")
print("=" * 60)

df = df_raw[df_raw['purpose'].str.lower().str.contains('sale', na=False)].copy()
print(f"  After keeping For Sale: {len(df):,} rows")

# ── Normalize location names ─────────────────────────────────────────────────
# Merges duplicates like "Federal B Area" and "Federal B Area, Karachi, Sindh"
# into a single canonical form, preventing fragmented price signals.
print("  Normalizing location names...")
df['location_raw'] = df['location'].copy()
df['location'] = df['location'].apply(normalize_location)
df['location'] = df['location'].replace('', pd.NA)
unique_before = df['location_raw'].nunique()
unique_after  = df['location'].nunique()
print(f"  Unique locations: {unique_before:,} -> {unique_after:,} (merged {unique_before - unique_after:,} duplicates)")

# ── Optional: Fuzzy deduplication pass ───────────────────────────────────────
# Catches remaining near-duplicates (genuine typos, one-off abbreviations)
# Only runs if rapidfuzz is installed; skips gracefully if not.
try:
    from rapidfuzz import process as fuzz_process, fuzz as _fuzz

    print("  Running fuzzy location deduplication (10-30 seconds)...")

    def fuzzy_deduplicate_locations(location_series, city_series, threshold=92):
        mapping = {}
        for city in city_series.dropna().unique():
            city_mask = city_series == city
            city_locs = location_series[city_mask].dropna()
            freq = city_locs.value_counts()
            unique_locs = freq.index.tolist()
            if len(unique_locs) < 2:
                continue
            for i in range(len(unique_locs) - 1, 0, -1):
                candidate = unique_locs[i]
                if candidate in mapping:
                    continue
                more_frequent = unique_locs[:i]
                match, score, _ = fuzz_process.extractOne(
                    candidate, more_frequent, scorer=_fuzz.token_set_ratio
                )
                if score >= threshold and abs(len(candidate) - len(match)) <= 20:
                    mapping[candidate] = match
        return mapping

    fuzzy_map = fuzzy_deduplicate_locations(df['location'], df['city'], threshold=92)
    if fuzzy_map:
        df['location'] = df['location'].map(lambda x: fuzzy_map.get(x, x) if pd.notna(x) else x)
        unique_after_fuzzy = df['location'].nunique()
        print(f"  Fuzzy dedup merged {len(fuzzy_map):,} additional variants -> {unique_after_fuzzy:,} unique locations")
        for src, dst in list(fuzzy_map.items())[:10]:
            print(f"    '{src}' -> '{dst}'")
    else:
        print("  Fuzzy dedup: no additional merges needed above threshold=92")

except ImportError:
    print("  rapidfuzz not installed — skipping fuzzy dedup (pip install rapidfuzz to enable)")
except Exception as fuzzy_err:
    print(f"  Fuzzy dedup failed ({fuzzy_err}) — continuing without it")

# Only train on cities with large, reliable datasets.
# Rawalpindi, Faisalabad, and Peshawar are excluded due to insufficient data volume
# which causes skewed and inaccurate predictions for those markets.
CITIES = ['lahore', 'karachi', 'islamabad']
df['city_lower'] = df['city'].str.lower().str.strip()
df = df[df['city_lower'].isin(CITIES)]
city_map = {
    'lahore':    'Lahore',
    'karachi':   'Karachi',
    'islamabad': 'Islamabad',
}
df['city'] = df['city_lower'].map(city_map)
print(f"  After city filter (Lahore / Karachi / Islamabad only): {len(df):,} rows")

KEEP_TYPES = ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'Farm House', 'Room', 'Penthouse']
df['property_type'] = df['property_type'].str.strip().str.title()
df = df[df['property_type'].isin(KEEP_TYPES)]
print(f"  After property type filter: {len(df):,} rows")

df = df.dropna(subset=['price', 'area_marla', 'city'])
print(f"  After dropping NaN critical cols: {len(df):,} rows")

df = df[(df['price'] >= 500_000) & (df['price'] <= 2_000_000_000)]
print(f"  After price range filter: {len(df):,} rows")

df = df[(df['area_marla'] >= 0.5) & (df['area_marla'] <= 10_000)]
print(f"  After area range filter: {len(df):,} rows")

df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['baths']    = pd.to_numeric(df['baths'], errors='coerce')
df['bedrooms'] = df.groupby('property_type')['bedrooms'].transform(lambda x: x.fillna(x.median()))
df['baths']    = df.groupby('property_type')['baths'].transform(lambda x: x.fillna(x.median()))
df['bedrooms'] = df['bedrooms'].fillna(2).clip(0, 20)
df['baths']    = df['baths'].fillna(2).clip(0, 20)

df = df.drop_duplicates(subset=['city', 'price', 'area_marla', 'bedrooms', 'baths'])
print(f"  After removing duplicates: {len(df):,} rows")

def remove_outliers_iqr(group):
    Q1 = group['price'].quantile(0.02)
    Q3 = group['price'].quantile(0.98)
    IQR = Q3 - Q1
    return group[(group['price'] >= Q1 - 3.0 * IQR) & (group['price'] <= Q3 + 3.0 * IQR)]

df = df.groupby('city', group_keys=False).apply(remove_outliers_iqr)
# pandas 2.2+ may drop the groupby key column from the result — restore it
df = df.reset_index(drop=True)
if 'city' not in df.columns:
    df['city'] = df['city_lower'].map(city_map)
print(f"  After outlier removal: {len(df):,} rows")

# ── Location normalization diagnostic ────────────────────────────────────────
print("\n  Top 10 locations by listing count (post-normalization):")
top_locs = df.groupby(['city', 'location'], dropna=True).size().sort_values(ascending=False).head(10)
for (city_name, loc), count in top_locs.items():
    print(f"    {city_name}: '{loc}' — {count:,} listings")

# ─── 3. FEATURE ENGINEERING ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Feature engineering...")
print("=" * 60)

df['area_sqyd'] = df['area_sqft'] / 9.0
df['log_price'] = np.log1p(df['price'])
df['log_area']  = np.log1p(df['area_marla'])

df['area_category_sqyd'] = df['area_sqyd'].apply(
    lambda x: 0 if x < 200 else (1 if x < 500 else (2 if x < 1000 else (3 if x < 2000 else 4)))
)

from sklearn.preprocessing import LabelEncoder
le_city  = LabelEncoder()
le_ptype = LabelEncoder()
df['city_encoded']  = le_city.fit_transform(df['city'].astype(str))
df['ptype_encoded'] = le_ptype.fit_transform(df['property_type'].astype(str))

# Location median price encoding (captures actual price level of neighbourhood)
loc_median_price    = df.groupby('location')['price'].median()
global_median_price = df['price'].median()
df['location_price_level']     = df['location'].map(loc_median_price).fillna(global_median_price)
df['log_location_price_level'] = np.log1p(df['location_price_level'])

# City × property_type interaction median price
city_ptype_median = df.groupby(['city', 'property_type'])['price'].median()
df['city_ptype_median_price'] = df.set_index(['city', 'property_type']).index.map(city_ptype_median)
df['city_ptype_median_price']  = df['city_ptype_median_price'].fillna(global_median_price)
df['log_city_ptype_median']    = np.log1p(df['city_ptype_median_price'])

print(f"  Features engineered. Final dataset: {len(df):,} rows")

# ─── 4. MODEL TRAINING ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training models...")
print("=" * 60)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    'log_area',
    'bedrooms',
    'baths',
    'city_encoded',
    'ptype_encoded',
    'log_location_price_level',
    'log_city_ptype_median',
    'area_category_sqyd',
]
FEATURE_NAMES = [
    'Log Area (Marla)',
    'Bedrooms',
    'Bathrooms',
    'City',
    'Property Type',
    'Location Price Level',
    'City × Type Median Price',
    'Size Category',
]

TARGET = 'log_price'
X = df[FEATURE_COLS].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train size: {len(X_train):,} | Test size: {len(X_test):,}")

models_to_try = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    ),
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
}

try:
    from xgboost import XGBRegressor
    models_to_try['XGBoost'] = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbosity=0
    )
    print("  XGBoost available — added to comparison")
except ImportError:
    print("  XGBoost not installed — skipping (pip install xgboost to enable)")

results = {}
best_model_name = None
best_r2 = -np.inf
best_model = None

for name, model in models_to_try.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_pkr = np.expm1(y_pred)
    y_test_pkr = np.expm1(y_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test_pkr, y_pred_pkr)
    rmse = np.sqrt(mean_squared_error(y_test_pkr, y_pred_pkr))

    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'model': model}
    print(f"    R2:   {r2:.4f}")
    print(f"    MAE:  PKR {mae:,.0f}")
    print(f"    RMSE: PKR {rmse:,.0f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

print(f"\n  Best model: {best_model_name} (R2 = {best_r2:.4f})")

# 5-fold cross-validation on full dataset
print(f"\n  Running 5-fold cross-validation on {best_model_name}...")
cv_scores = None
try:
    from sklearn.model_selection import cross_val_score as _cv_score
    cv_scores = _cv_score(best_model, X, y, cv=5, scoring='r2', n_jobs=1)
    print(f"  Cross-validated R2 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
except Exception as cv_err:
    print(f"  Cross-validation skipped ({type(cv_err).__name__}: {cv_err})")

feature_importances = None
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
elif hasattr(best_model, 'steps'):
    inner = best_model.named_steps.get('model', None)
    if hasattr(inner, 'feature_importances_'):
        feature_importances = inner.feature_importances_
    elif hasattr(inner, 'coef_'):
        feature_importances = np.abs(inner.coef_)

# ─── 5. SAVE MODEL ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving model...")
print("=" * 60)

y_pred_best     = best_model.predict(X_test)
residuals_log   = y_test - y_pred_best
y_pred_pkr_best = np.expm1(y_pred_best)
y_test_pkr_arr  = np.expm1(y_test)

city_stats = df.groupby(['city', 'property_type']).agg(
    avg_price=('price', 'median'),
    count=('price', 'count'),
    avg_marla=('area_marla', 'median'),
).reset_index()

area_cat_stats = df.groupby(['city', 'property_type', 'area_category_sqyd']).agg(
    avg_price=('price', 'median'),
    count=('price', 'count'),
).reset_index()

AREA_CAT_LABELS = {
    0: "< 200 Sq. Yd.",
    1: "200–500 Sq. Yd.",
    2: "500–1000 Sq. Yd.",
    3: "1000–2000 Sq. Yd.",
    4: "> 2000 Sq. Yd."
}
area_cat_stats['area_label'] = area_cat_stats['area_category_sqyd'].map(AREA_CAT_LABELS)

payload = {
    # Model & encoders
    'model':                    best_model,
    'model_name':               best_model_name,
    'label_encoder_city':       le_city,
    'label_encoder_ptype':      le_ptype,
    # Location encoding maps
    'location_median_price_map': loc_median_price.to_dict(),
    'global_median_price':       global_median_price,
    'city_ptype_median_map':     city_ptype_median.to_dict(),
    # Feature metadata
    'feature_cols':             FEATURE_COLS,
    'feature_names':            FEATURE_NAMES,
    'feature_importances':      feature_importances,
    # Model results
    'results': {
        name: {'R2': v['R2'], 'MAE': v['MAE'], 'RMSE': v['RMSE']}
        for name, v in results.items()
    },
    'cv_scores':               cv_scores.tolist() if cv_scores is not None else None,
    # Diagnostics
    'y_test_log':              y_test,
    'y_pred_log':              y_pred_best,
    'y_test_pkr':              y_test_pkr_arr,
    'y_pred_pkr':              y_pred_pkr_best,
    'residuals':               residuals_log,
    # Market data
    'df_clean':                df[['city','location','property_type','bedrooms','baths',
                                   'price','area_marla','area_sqft','area_sqyd']].copy(),
    'city_stats':              city_stats,
    'area_cat_stats':          area_cat_stats,
    'area_cat_labels':         AREA_CAT_LABELS,
    # Metadata
    'total_records':           len(df),
    'cities_supported':        sorted(df['city'].unique().tolist()),
    'property_types':          sorted(df['property_type'].unique().tolist()),
    'city_locations_map':      {
        city: [loc.title() for loc in sorted(group['location'].dropna().unique().tolist())]
        for city, group in df.groupby('city')
    },
    'locations_are_normalized': True,
}

joblib.dump(payload, MODEL_PATH)
print(f"  Model saved to: {MODEL_PATH}")
print(f"  Total records used: {len(df):,}")
print("\n" + "=" * 60)
print("TRAINING COMPLETE! Now run: streamlit run app.py")
print("=" * 60)
