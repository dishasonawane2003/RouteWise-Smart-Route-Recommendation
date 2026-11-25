"""
RouteWise - Traffic Prediction & Clustering Module
Academic Level: M.Tech First Year (Mini-Project)
Team: 3 Members | City: Pune, Maharashtra

This module handles:
1. Generation of simulated Pune traffic dataset (clearly labeled as simulated).
2. Data preprocessing.
3. Training and evaluation of:
   - K-Nearest Neighbors (KNN) -> Congestion classification (Low, Medium, High)
   - Random Forest Regressor -> Travel time prediction (in minutes)
   - K-Means Clustering -> Grouping traffic conditions into congestion clusters
4. Model serialization using joblib into the `models/` directory.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
import joblib

DATASET_PATH = "traffic_data.csv"
MODEL_DIR = "models"

# Key Pune arterial locations
PUNE_NODES = [
    "Shivajinagar", "Kothrud", "Hinjewadi", "Viman Nagar", "Hadapsar",
    "Swargate", "Deccan Gymkhana", "Baner", "Aundh", "Katraj",
    "Pimpri", "Wakad", "Kalyani Nagar", "Koregaon Park", "Magarpatta"
]


def generate_pune_traffic_dataset(num_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, simulated traffic dataset representing major Pune corridors.
    NOTE: This is a clearly labelled simulated academic dataset based on typical
    urban Pune traffic patterns (peak rush hours 9-11 AM and 6-9 PM on weekdays).
    """
    np.random.seed(random_state)

    sources = np.random.choice(PUNE_NODES, size=num_samples)
    destinations = []
    for s in sources:
        candidates = [n for n in PUNE_NODES if n != s]
        destinations.append(np.random.choice(candidates))

    # Features:
    # 1. Distance between 3.0 km and 32.0 km (Pune urban sprawl scale)
    distance_km = np.round(np.random.uniform(3.0, 32.0, size=num_samples), 2)

    # 2. Hour of the day (0 to 23)
    # Give higher probability to daytime hours (7 to 22)
    hour_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06,  # 00:00 - 07:00
        0.08, 0.09, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05,  # 08:00 - 15:00
        0.06, 0.08, 0.09, 0.08, 0.05, 0.03, 0.02, 0.01   # 16:00 - 23:00
    ])
    hour_probs = hour_probs / hour_probs.sum()
    hour_of_day = np.random.choice(np.arange(24), size=num_samples, p=hour_probs)

    # 3. Day of week (0: Monday ... 6: Sunday)
    day_of_week = np.random.choice(np.arange(7), size=num_samples)
    is_weekend = np.where(day_of_week >= 5, 1, 0)

    # Determine traffic condition based on realistic Pune peak-hour dynamics
    # Peak hours: Morning (8:30 - 11:30) and Evening (17:30 - 21:00) on weekdays
    is_weekday_peak = ((hour_of_day >= 9) & (hour_of_day <= 11) | (hour_of_day >= 18) & (hour_of_day <= 21)) & (is_weekend == 0)
    is_moderate_hours = ((hour_of_day >= 12) & (hour_of_day <= 17) | (hour_of_day >= 7) & (hour_of_day <= 8))
    is_night_hours = (hour_of_day >= 22) | (hour_of_day <= 6)

    avg_speed_kmh = []
    congestion_levels = []

    for i in range(num_samples):
        # Base speed with some stochastic variance
        if is_weekday_peak[i]:
            speed = np.random.normal(16.0, 4.0)  # Heavy traffic (e.g. Hinjewadi bridge / SB Road)
            speed = max(8.0, min(speed, 25.0))
            congestion = "High" if speed < 18 else "Medium"
        elif is_moderate_hours[i]:
            speed = np.random.normal(28.0, 5.0)
            speed = max(18.0, min(speed, 40.0))
            congestion = "Medium" if speed < 30 else "Low"
        elif is_night_hours[i]:
            speed = np.random.normal(46.0, 6.0)
            speed = max(32.0, min(speed, 65.0))
            congestion = "Low"
        else:
            # Weekend / non-peak regular hours
            speed = np.random.normal(32.0, 6.0)
            speed = max(20.0, min(speed, 48.0))
            congestion = "Low" if speed > 28 else "Medium"

        avg_speed_kmh.append(round(speed, 1))
        congestion_levels.append(congestion)

    avg_speed_kmh = np.array(avg_speed_kmh)
    
    # Calculate travel time in minutes: (distance / speed) * 60 + signal/stop delay
    # High congestion adds extra signal waiting minutes
    signal_delays = np.where(
        np.array(congestion_levels) == "High",
        np.random.uniform(8.0, 18.0, size=num_samples),
        np.where(
            np.array(congestion_levels) == "Medium",
            np.random.uniform(3.0, 8.0, size=num_samples),
            np.random.uniform(1.0, 4.0, size=num_samples)
        )
    )
    travel_time_min = np.round(((distance_km / avg_speed_kmh) * 60) + signal_delays, 1)

    df = pd.DataFrame({
        "source": sources,
        "destination": destinations,
        "distance_km": distance_km,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "avg_speed_kmh": avg_speed_kmh,
        "congestion_level": congestion_levels,
        "travel_time_min": travel_time_min,
        "dataset_type": "SIMULATED_PUNE_CORRIDORS"
    })

    return df


def train_and_save_models():
    """
    Trains KNN, Random Forest, and K-Means models and evaluates their performance.
    Saves trained artifacts in the models/ folder.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Generating simulated Pune traffic dataset...")
    df = generate_pune_traffic_dataset(num_samples=1600, random_state=42)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset successfully saved to {DATASET_PATH} ({len(df)} records).")

    # Feature definitions
    # Congestion prediction features
    feature_cols_knn = ["distance_km", "hour_of_day", "day_of_week", "is_weekend", "avg_speed_kmh"]
    X_knn = df[feature_cols_knn]
    y_knn = df["congestion_level"]

    # Encode target labels: High: 0, Medium: 1, Low: 2
    label_encoder = LabelEncoder()
    y_knn_encoded = label_encoder.fit_transform(y_knn)

    # Train/Test Split
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_knn, y_knn_encoded, test_size=0.2, random_state=42, stratify=y_knn_encoded
    )

    # Standard Scaler for distance and speed
    scaler_knn = StandardScaler()
    X_train_k_scaled = scaler_knn.fit_transform(X_train_k)
    X_test_k_scaled = scaler_knn.transform(X_test_k)

    # 1. K-Nearest Neighbors (KNN) Classifier
    print("\n" + "="*50)
    print("Training 1: K-Nearest Neighbors (KNN) for Traffic Congestion")
    print("="*50)
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_model.fit(X_train_k_scaled, y_train_k)
    y_pred_k = knn_model.predict(X_test_k_scaled)

    acc = accuracy_score(y_test_k, y_pred_k)
    print(f"KNN Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test_k, y_pred_k, target_names=label_encoder.classes_))

    # 2. Random Forest Regressor for Travel Time Prediction
    print("="*50)
    print("Training 2: Random Forest Regressor for Travel Time (Minutes)")
    print("="*50)
    # Features include distance, hour, day, weekend, and encoded congestion level
    df["congestion_encoded"] = label_encoder.transform(df["congestion_level"])
    feature_cols_rf = ["distance_km", "hour_of_day", "day_of_week", "is_weekend", "congestion_encoded", "avg_speed_kmh"]
    X_rf = df[feature_cols_rf]
    y_rf = df["travel_time_min"]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_model.predict(X_test_rf)

    mae = mean_absolute_error(y_test_rf, y_pred_rf)
    rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    r2 = r2_score(y_test_rf, y_pred_rf)

    print(f"Random Forest Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"Random Forest Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    print(f"Random Forest R² Score: {r2:.4f}")

    # 3. K-Means Clustering for Traffic Patterns
    print("="*50)
    print("Training 3: K-Means Clustering for Congestion Grouping")
    print("="*50)
    # Group observations by [hour_of_day, avg_speed_kmh, travel_time_min]
    cluster_features = ["hour_of_day", "avg_speed_kmh", "travel_time_min"]
    X_cluster = df[cluster_features]
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(X_cluster_scaled)
    sil_score = silhouette_score(X_cluster_scaled, clusters)
    print(f"K-Means Silhouette Score (k=3): {sil_score:.4f}")

    # Interpret clusters by mean values
    df["cluster"] = clusters
    cluster_summary = df.groupby("cluster")[cluster_features].mean()
    print("\nCluster Center Interpretations (Mean Values):")
    print(cluster_summary)

    # Save models, scaler, and label encoder
    print("\nSaving trained models to models/ ...")
    joblib.dump(knn_model, os.path.join(MODEL_DIR, "knn.pkl"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(kmeans_model, os.path.join(MODEL_DIR, "kmeans.pkl"))
    joblib.dump(scaler_knn, os.path.join(MODEL_DIR, "scaler_knn.pkl"))
    joblib.dump(scaler_cluster, os.path.join(MODEL_DIR, "scaler_cluster.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print("All models and scalers successfully saved!")


class TrafficPredictor:
    """
    Inference class loaded by the Streamlit application for real-time predictions.
    """
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.knn = joblib.load(os.path.join(model_dir, "knn.pkl"))
        self.rf = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
        self.kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
        self.scaler_knn = joblib.load(os.path.join(model_dir, "scaler_knn.pkl"))
        self.scaler_cluster = joblib.load(os.path.join(model_dir, "scaler_cluster.pkl"))
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    def estimate_speed_by_context(self, hour: int, day_of_week: int) -> float:
        """Heuristic speed estimation based on time in Pune."""
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak = ((hour >= 9 and hour <= 11) or (hour >= 18 and hour <= 21)) and (not is_weekend)
        if is_peak:
            return 17.5  # Typical peak crawl
        elif (hour >= 23 or hour <= 6):
            return 45.0  # Night open roads
        else:
            return 28.0  # Regular daytime flow

    def predict_congestion(self, distance_km: float, hour: int, day_of_week: int, avg_speed: float = None) -> dict:
        """
        Uses KNN to classify congestion level as Low, Medium, or High.
        Returns label and class probabilities.
        """
        if avg_speed is None:
            avg_speed = self.estimate_speed_by_context(hour, day_of_week)

        is_weekend = 1 if day_of_week >= 5 else 0
        input_data = pd.DataFrame([{
            "distance_km": distance_km,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "avg_speed_kmh": avg_speed
        }])

        scaled_input = self.scaler_knn.transform(input_data)
        encoded_pred = self.knn.predict(scaled_input)[0]
        probabilities = self.knn.predict_proba(scaled_input)[0]
        
        congestion_label = self.label_encoder.inverse_transform([encoded_pred])[0]
        prob_dict = {label: round(float(prob), 3) for label, prob in zip(self.label_encoder.classes_, probabilities)}

        return {
            "congestion_level": congestion_label,
            "encoded_level": int(encoded_pred),
            "estimated_speed_kmh": avg_speed,
            "probabilities": prob_dict
        }

    def predict_travel_time(self, distance_km: float, hour: int, day_of_week: int, congestion_level: str = None) -> float:
        """
        Uses Random Forest to predict travel time in minutes.
        """
        is_weekend = 1 if day_of_week >= 5 else 0
        if congestion_level is None:
            c_info = self.predict_congestion(distance_km, hour, day_of_week)
            congestion_level = c_info["congestion_level"]
            avg_speed = c_info["estimated_speed_kmh"]
        else:
            avg_speed = self.estimate_speed_by_context(hour, day_of_week)

        encoded_congestion = self.label_encoder.transform([congestion_level])[0]

        input_data = pd.DataFrame([{
            "distance_km": distance_km,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "congestion_encoded": encoded_congestion,
            "avg_speed_kmh": avg_speed
        }])

        predicted_minutes = self.rf.predict(input_data)[0]
        return round(float(predicted_minutes), 1)

    def analyze_traffic_cluster(self, hour: int, avg_speed: float, travel_time: float) -> dict:
        """
        Assigns route observation to one of 3 K-Means clusters and provides interpretation.
        """
        input_data = pd.DataFrame([{
            "hour_of_day": hour,
            "avg_speed_kmh": avg_speed,
            "travel_time_min": travel_time
        }])
        scaled = self.scaler_cluster.transform(input_data)
        cluster_id = int(self.kmeans.predict(scaled)[0])

        cluster_descriptions = {
            0: "Cluster A: Free Flow / Off-Peak (Higher speed, minimal transit delay)",
            1: "Cluster B: Moderate Urban Transit (Standard daytime speed with minor bottlenecks)",
            2: "Cluster C: Severe Congestion / Peak Rush (Low vehicle speed, prolonged waiting delays)"
        }
        return {
            "cluster_id": cluster_id,
            "description": cluster_descriptions.get(cluster_id, "Cluster Group")
        }


if __name__ == "__main__":
    train_and_save_models()
