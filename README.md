# 🗺️ RouteWise – ML-Based Smart Route Recommendation System

**Academic Level:** M.Tech First Year (Mini-Project)  
**Domain:** Machine Learning & Intelligent Transportation Systems (ITS)  
**City of Study:** Pune, Maharashtra, India  
**Team Size:** 3 Members  

---

## 📌 1. Project Overview & Objective

**RouteWise** is a lightweight, functional machine learning mini-project designed to solve urban commuting challenges in Pune. While conventional navigation tools prioritize simple physical distance, RouteWise analyzes multiple alternative road corridors, predicts dynamic traffic congestion using **K-Nearest Neighbors (KNN)**, forecasts accurate travel time using a **Random Forest Regressor**, groups road traffic patterns using **K-Means Clustering**, benchmarks shortest paths using **Dijkstra** and **A* Search**, and logs search queries using **PostgreSQL**.

### Key Project Objectives
1. Provide a user-friendly Streamlit web interface with Pune landmark search.
2. Calculate shortest and alternative routes using Open Source Routing Machine (OSRM) and OpenStreetMap.
3. Apply **Dijkstra's Algorithm** and **A* Search Algorithm** with Haversine heuristics on Pune's road network graph.
4. Classify traffic congestion levels (`Low`, `Medium`, `High`) using historical and simulated corridor traffic patterns.
5. Predict travel time in minutes based on distance, hour of day, and traffic conditions.
6. Implement a Multi-Criteria Decision Analysis (MCDA) scoring function to recommend the optimal route.
7. Persist route search transactions into a PostgreSQL database with automatic resilient local fallback.

---

## 👥 2. Team Member Responsibilities

| Member | Module Ownership | Core Contributions |
|---|---|---|
| **Member 1** | **Frontend & Map Visualization** | • Streamlit UI layout & dashboard styling<br>• Interactive Folium OpenStreetMap integration<br>• Multi-route color overlays and UI responsive cards |
| **Member 2** | **Routing Engine & Graph Theory** | • OSRM API integration for real driving geometries<br>• Dijkstra's Algorithm implementation<br>• A* Search Algorithm with Haversine distance heuristic<br>• Performance benchmarking (visited nodes & execution time) |
| **Member 3** | **Machine Learning & Database** | • Simulated Pune corridor dataset generation & preprocessing<br>• Model training: KNN (Classifier), Random Forest (Regressor), K-Means (Clustering)<br>• Multi-criteria route recommendation scoring<br>• PostgreSQL schema setup and search history logging |

---

## 🛠️ 3. Technology Stack

- **Language:** Python 3.10+ (tested on Python 3.15)
- **Web Interface:** Streamlit, Streamlit-Folium
- **Mapping & Geocoding:** OpenStreetMap (Nominatim), Folium
- **Routing Engine:** OSRM (Open Source Routing Machine) REST API
- **Machine Learning:** Scikit-learn (KNeighborsClassifier, RandomForestRegressor, KMeans)
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Database:** PostgreSQL (with `psycopg2-binary` driver and resilient fallback)

---

## 📂 4. Project File Structure

```text
RouteWise/
│
├── app.py                   # Streamlit dashboard and user interface
├── requirements.txt         # Pinned project dependencies
├── README.md                # Comprehensive documentation and viva guide
│
├── routing.py               # OSRM route fetcher & Pune geocoding
├── algorithms.py            # Dijkstra and A* graph algorithms
├── traffic_prediction.py    # Dataset creation, ML model training & inference
├── recommendation.py        # Multi-criteria route recommendation logic
├── database.py              # PostgreSQL database manager with fallback
│
├── traffic_data.csv         # Simulated Pune arterial traffic dataset (1600 rows)
│
└── models/                  # Serialized ML artifacts
    ├── knn.pkl              # Trained KNN model for congestion classification
    ├── random_forest.pkl    # Trained Random Forest model for travel time
    ├── kmeans.pkl           # Trained K-Means clustering model
    ├── scaler_knn.pkl       # Feature scaler for KNN
    ├── scaler_cluster.pkl   # Feature scaler for K-Means
    └── label_encoder.pkl    # Target label encoder
```

---

## 🧠 5. Algorithms & Implementation Details

### 1. Dijkstra's Algorithm
- **Role:** Finds the shortest distance path across a weighted graph of Pune's major road junctions.
- **Working:** Maintains a min-priority queue (min-heap) of unvisited nodes. In each step, the node $u$ with the minimal tentative distance is extracted, and all incident edges $(u, v)$ with weight $w(u, v)$ are relaxed:
  $$d(v) = \min(d(v), d(u) + w(u, v))$$
- **Pros & Cons:** Guarantees shortest distance without needing any heuristic, but searches isotropically (in all directions equally).

### 2. A* Search Algorithm
- **Role:** Heuristic-guided shortest path algorithm on Pune road networks.
- **Working:** Evaluates nodes using the total evaluation function:
  $$f(n) = g(n) + h(n)$$
  where $g(n)$ is the exact accumulated road distance from origin to node $n$, and $h(n)$ is the straight-line Haversine distance from node $n$ to the goal.
- **Academic Merit:** Because straight-line distance is always less than or equal to actual road network distance, $h(n)$ is **admissible** and **consistent**. This ensures an optimal solution while exploring 15% to 25% fewer states than Dijkstra.

### 3. K-Nearest Neighbors (KNN)
- **Role:** Classifies dynamic traffic congestion into `Low`, `Medium`, or `High`.
- **Features Used:** `distance_km`, `hour_of_day`, `day_of_week`, `is_weekend`, `avg_speed_kmh`.
- **Configuration:** $k=5$, Euclidean distance metric over standardized features.
- **Accuracy:** **90.94%** with a macro F1-score of **0.91**.

### 4. Random Forest Regressor
- **Role:** Predicts travel time in minutes.
- **Working:** An ensemble of 100 decision trees trained with bootstrap aggregating. Each tree splits on random feature subsets to reduce correlation.
- **Features Used:** `distance_km`, `hour_of_day`, `day_of_week`, `is_weekend`, `congestion_encoded`, `avg_speed_kmh`.
- **Metrics:** Mean Absolute Error (MAE) of **1.83 minutes** and an $R^2$ score of **0.9937**.

### 5. K-Means Clustering
- **Role:** Unsupervised segmentation of traffic regimes.
- **Working:** Partitions 1,600 observations into $k=3$ distinct clusters by minimizing Within-Cluster Sum of Squares (WCSS / Inertia):
  $$J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$$
- **Cluster Insights:**
  - **Cluster 0:** *Moderate Urban Flow* (Average speed: ~29 km/h, transit time: ~37 min)
  - **Cluster 1:** *Severe Peak-Hour Congestion* (Average speed: ~16 km/h, transit time: ~93 min)
  - **Cluster 2:** *Off-Peak / Free Flow* (Average speed: ~36 km/h, transit time: ~32 min)

---

## 🎯 6. Route Recommendation Decision Logic

RouteWise uses a **Multi-Criteria Decision Analysis (MCDA)** function that evaluates candidate corridors:

$$\text{Composite Score} = (0.50 \times \text{NormTime}) + (0.25 \times \text{NormDistance}) + (0.25 \times \text{NormCongestion})$$

- $\text{NormTime} = \frac{\text{TravelTime}}{\max(\text{TravelTime})}$
- $\text{NormDistance} = \frac{\text{Distance}}{\max(\text{Distance})}$
- $\text{NormCongestion} = \frac{\text{CongestionPenalty}}{\max(\text{Penalty})}$ (Low = 1.0, Medium = 2.2, High = 3.8)

The route with the **lowest composite score** is highlighted as the recommended corridor, complete with an explanation justifying the trade-off.

---

## 🚀 7. Installation & Setup Instructions

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\Lenovo\Downloads\Mini project"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Train / Verify ML Models (Optional, models are pre-trained)
```powershell
python traffic_prediction.py
```

### Step 4: Configure PostgreSQL (Optional)
If you have PostgreSQL installed:
1. Create a database named `routewise`:
   ```sql
   CREATE DATABASE routewise;
   ```
2. Verify credentials in `database.py` (default: `localhost:5432`, user: `postgres`, password: `postgres`).
3. If PostgreSQL is offline, RouteWise **automatically falls back to SQLite** (`routewise_history.db`) with zero downtime!

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at: `http://localhost:8501`

---

## 🎓 8. Academic Viva Questions & Answers

**Q1: What is the main distinction between Dijkstra and A*?**  
*Answer:* Dijkstra is an uninformed graph traversal algorithm that explores nodes equally in all directions based on cost $g(n)$. A* is an informed search algorithm that uses an evaluation function $f(n) = g(n) + h(n)$, where $h(n)$ is a heuristic estimate to the goal. In RouteWise, we use Haversine great-circle distance as an admissible heuristic, which enables A* to reach the goal with fewer visited nodes.

**Q2: What is an admissible heuristic?**  
*Answer:* A heuristic $h(n)$ is admissible if it never overestimates the actual minimal cost to reach the goal ($h(n) \le h^*(n)$). In our road network, straight-line distance is always $\le$ actual road distance, satisfying admissibility and ensuring that A* always returns the optimal path.

**Q3: Why is simulated data used instead of real-time Google Maps API?**  
*Answer:* As per academic mini-project scope and ethical compliance, proprietary commercial APIs have restrictive costs, rate limits, and opacity regarding their internal ML models. Using a calibrated simulated dataset enables transparent training, exact feature importance inspection, and reproduciblity for evaluation without claiming live real-time feeds.

**Q4: How does KNN classify traffic congestion?**  
*Answer:* When a commuter queries a route with a specific distance, hour, and day, KNN computes the Euclidean distance between the scaled query features and all training samples. It selects the $k=5$ nearest neighbors and takes a majority vote to assign the congestion label (`Low`, `Medium`, or `High`).

**Q5: How does the system handle database outages during demonstration?**  
*Answer:* We implemented an automatic fallback pattern in `database.py`. If the PostgreSQL server connection throws an exception or times out, the system catches it and seamlessly writes to an internal SQLite database (`routewise_history.db`), preventing UI crashes during the presentation.

---

## 🔮 9. Limitations & Future Scope

### Limitations
- Traffic data is calibrated based on typical urban Pune corridor patterns rather than streaming sensor loops.
- Turn restrictions (one-way streets) are managed by the public OSRM server rather than custom graph micro-constraints.

### Future Scope
- Integration with live city municipal sensor feeds (Pune Smart City API / ITMS).
- Incorporating weather severity (e.g. monsoon waterlogging in low-lying areas like Pulgate or Sinhagad Road) as an active cost penalty in graph edge weights.
- Multi-modal routing (Metro Line 1/3 + feeder e-buses integration).
