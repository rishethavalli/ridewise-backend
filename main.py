from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os

app = FastAPI()

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and features with robust fallbacks
def safe_load_pickle(path: str):
    """Load pickle files saved in different Python versions.

    If unpickling fails, return None so we can fall back to heuristic predictions
    instead of crashing the server.
    """
    # Convert to absolute path if not already
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    
    # Try multiple loading strategies for compatibility
    strategies = [
        lambda f: pickle.load(f),  # Default
        lambda f: pickle.load(f, encoding="latin1"),  # Python 2 compatibility
        lambda f: pickle.load(f, encoding="bytes"),  # Bytes mode
        lambda f: pickle.load(f, fix_imports=True, encoding="latin1"),  # With fix_imports
    ]
    
    for strategy in strategies:
        try:
            with open(path, "rb") as f:
                return strategy(f)
        except Exception as e:
            continue
    
    # If all strategies fail, warn and return None
    print(f"[WARN] Failed to load pickle {os.path.basename(path)} with all strategies. Using fallback predictions.")
    return None


day_model = safe_load_pickle("day_model.pkl")
hour_model = safe_load_pickle("hour_model.pkl")
day_features = safe_load_pickle("day_features.pkl") or []
hour_features = safe_load_pickle("hour_features.pkl") or []

# Debug: Check feature mismatch
print(f"[INFO] day_features: {len(day_features)} features")
print(f"[INFO] hour_features: {len(hour_features)} features")

class PredictInput(BaseModel):
    season: str
    month: int
    day_of_week: int
    hour: int
    temp: float
    hum: float
    wind: float
    weather: str
    holiday: int
    workingday: int


def fallback_prediction(data: PredictInput, mode: str) -> int:
    """Heuristic fallback matching the frontend's local calculation."""
    base_prediction = 4500 if mode == "daily" else 180
    season_multiplier = {"spring": 1.1, "summer": 1.3, "fall": 1.0, "winter": 0.7}
    weather_multiplier = {"clear": 1.2, "cloudy": 1.0, "rain": 0.7, "storm": 0.4}

    predicted = (
        base_prediction
        * season_multiplier.get(data.season, 1.0)
        * weather_multiplier.get(data.weather, 1.0)
        * (data.temp / 20.0)
        * (1.2 if data.workingday else 0.8)
        * (0.9 if data.holiday else 1.0)
    )
    return int(round(predicted))

def preprocess(data: PredictInput, features: list, target_size=None):
    """Create feature vector matching the model's expected features."""
    # Map season and weather to numeric
    season_map = {"spring": 2, "summer": 3, "fall": 4, "winter": 1}
    weather_map = {"clear": 1, "cloudy": 2, "rain": 3, "storm": 4}
    
    # Build feature dict with all possible features
    feature_dict = {
        'yr': 0,  # Assuming recent year
        'mnth': data.month,
        'holiday': data.holiday,
        'workingday': data.workingday,
        'temp': data.temp / 50.0,  # Normalize to 0-1 range
        'atemp': data.temp / 50.0,  # Use temp as apparent temp
        'hum': data.hum / 100.0,  # Normalize to 0-1
        'windspeed': data.wind / 40.0,  # Normalize to 0-1
        'day_of_week': data.day_of_week,
        'month': data.month,
        'year': 2024,  # Assuming current year
        'cnt_lag_1': 4500,  # Average historical value
        'cnt_lag_7': 4500,
        'cnt_lag_14': 4500,
        'cnt_roll_7': 4500,
        'temp_hum': (data.temp / 50.0) * (data.hum / 100.0),
        'temp_wind': (data.temp / 50.0) * (data.wind / 40.0),
        # One-hot encoded features
        'season_2': 1 if season_map.get(data.season, 1) == 2 else 0,
        'season_3': 1 if season_map.get(data.season, 1) == 3 else 0,
        'season_4': 1 if season_map.get(data.season, 1) == 4 else 0,
        'weathersit_2': 1 if weather_map.get(data.weather, 1) == 2 else 0,
        'weathersit_3': 1 if weather_map.get(data.weather, 1) >= 3 else 0,
        'weekday_1': 1 if data.day_of_week == 1 else 0,
        'weekday_2': 1 if data.day_of_week == 2 else 0,
        'weekday_3': 1 if data.day_of_week == 3 else 0,
        'weekday_4': 1 if data.day_of_week == 4 else 0,
        'weekday_5': 1 if data.day_of_week == 5 else 0,
        'weekday_6': 1 if data.day_of_week == 6 else 0,
    }
    
    # Extract values in the order expected by the model
    values = [feature_dict.get(f, 0) for f in features]
    arr = np.array(values).reshape(1, -1)
    
    # Pad with zeros if we need more features than available
    if target_size and arr.shape[1] < target_size:
        padding = np.zeros((arr.shape[0], target_size - arr.shape[1]))
        arr = np.hstack([arr, padding])
    
    return arr

@app.post("/predict/day")
def predict_day(data: PredictInput):
    if day_model is None or not day_features:
        return {"count": fallback_prediction(data, "daily"), "source": "fallback"}
    
    try:
        # Use day_features (28 features) for day model
        X = preprocess(data, day_features, target_size=len(day_features))
        print(f"[DEBUG] Day model: shape {X.shape}, features: {len(day_features)}")
        pred = day_model.predict(X)[0]
        return {"count": int(pred), "source": "model"}
    except Exception as e:
        print(f"[ERROR] Day prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return {"count": fallback_prediction(data, "daily"), "source": "fallback"}

@app.post("/predict/hour")
def predict_hour(data: PredictInput):
    if hour_model is None or not hour_features:
        return {"count": fallback_prediction(data, "hourly"), "source": "fallback"}

    try:
        X = preprocess(data, hour_features)
        pred = hour_model.predict(X)[0]
        return {"count": int(pred), "source": "model"}
    except Exception as e:
        print(f"[ERROR] Hour prediction failed: {e}")
        return {"count": fallback_prediction(data, "hourly"), "source": "fallback"}

@app.get("/")
def root():
    return {"status": "RideWise backend running"}
