"""
RouteWise: ML-Based Smart Route Recommendation System
Academic Level: M.Tech First Year (Mini-Project)
Team: 3 Members | City: Pune, Maharashtra

Main Streamlit Application
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime

# Local project imports
from algorithms import PUNE_ROAD_NETWORK, PuneGraph, dijkstra_algorithm, astar_algorithm, compare_algorithms
from routing import search_pune_location, fetch_osrm_routes, PUNE_LANDMARKS
from traffic_prediction import TrafficPredictor
from recommendation import RouteRecommender
from database import DatabaseManager

# Page Configuration
st.set_page_config(
    page_title="RouteWise - Smart Route Recommendation",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_modules():
    """Initializes and caches heavy backend models and database connections."""
    predictor = TrafficPredictor()
    recommender = RouteRecommender(predictor=predictor)
    db = DatabaseManager()
    graph = PuneGraph()
    return predictor, recommender, db, graph


predictor, recommender, db, graph = load_system_modules()

# Header Section
st.markdown('<div class="main-title">🗺️ RouteWise</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title"><b>ML-Based Smart Route Recommendation System</b> &nbsp;|&nbsp; '
    'M.Tech Mini-Project &nbsp;|&nbsp; Pune, Maharashtra</div>',
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Settings")
st.sidebar.caption("Control temporal context to simulate real-world Pune traffic variations.")

sim_hour = st.sidebar.slider("⏰ Hour of Day (24-Hour)", min_value=0, max_value=23, value=datetime.now().hour)
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
today_idx = datetime.now().weekday()
sim_day_str = st.sidebar.selectbox("📅 Day of the Week", options=day_options, index=today_idx)
sim_day = day_options.index(sim_day_str)

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Project Team")
st.sidebar.markdown("""
- **Member 1**: Streamlit UI & Interactive Map (Folium)
- **Member 2**: OSRM Routing, Dijkstra & A* Graph Algorithms
- **Member 3**: ML Models (KNN, Random Forest, K-Means) & PostgreSQL
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Database Status")
st.sidebar.info(f"**Backend:** `{db.backend}`")
if "Fallback" in db.backend:
    st.sidebar.caption("ℹ️ PostgreSQL is currently in fallback mode. Search history is preserved locally.")


# Navigation Tabs
tab_navigate, tab_algorithms, tab_history, tab_academic = st.tabs([
    "🧭 Route Finder & Recommendation",
    "⚡ Dijkstra vs A* Graph Analysis",
    "📜 Search History (PostgreSQL)",
    "🎓 Academic Documentation & Viva Prep"
])


# ==========================================
# TAB 1: Route Finder & Recommendation
# ==========================================
with tab_navigate:
    col_input1, col_input2 = st.columns(2)
    
    locations_list = sorted(list(PUNE_LANDMARKS.keys()))
    default_src_idx = locations_list.index("Hinjewadi") if "Hinjewadi" in locations_list else 0
    default_dst_idx = locations_list.index("Hadapsar") if "Hadapsar" in locations_list else 1

    with col_input1:
        source_selection = st.selectbox("📍 Source Location", options=locations_list, index=default_src_idx)
        custom_src = st.text_input("Or enter custom Pune source address:", placeholder="e.g. Pune Station, Kothrud")
        source_query = custom_src.strip() if custom_src.strip() else source_selection

    with col_input2:
        dest_selection = st.selectbox("🎯 Destination Location", options=locations_list, index=default_dst_idx)
        custom_dst = st.text_input("Or enter custom Pune destination address:", placeholder="e.g. Phoenix Marketcity, Viman Nagar")
        dest_query = custom_dst.strip() if custom_dst.strip() else dest_selection

    find_route_btn = st.button("🚀 Find & Recommend Best Route", type="primary", use_container_width=True)

    if find_route_btn:
        if source_query.lower() == dest_query.lower():
            st.error("Source and destination cannot be identical. Please select different points.")
        else:
            with st.spinner("Geocoding locations and querying OSRM road geometry..."):
                src_res = search_pune_location(source_query)
                dst_res = search_pune_location(dest_query)

            if not src_res or not dst_res:
                st.error("Could not resolve coordinates for the specified Pune locations. Please try standard landmarks.")
            else:
                src_lat, src_lon, src_name = src_res
                dst_lat, dst_lon, dst_name = dst_res

                st.session_state["route_computed"] = True
                st.session_state["src_info"] = (src_lat, src_lon, src_name)
                st.session_state["dst_info"] = (dst_lat, dst_lon, dst_name)

                # Fetch OSRM routes
                routes = fetch_osrm_routes((src_lat, src_lon), (dst_lat, dst_lon))
                
                # ML Multi-Criteria Evaluation
                eval_result = recommender.evaluate_routes(routes, sim_hour, sim_day)
                st.session_state["eval_result"] = eval_result

                # Save recommended search to Database
                best_route = next(r for r in eval_result["routes"] if r["route_id"] == eval_result["recommended_route_id"])
                db.save_search(
                    source=src_name,
                    destination=dst_name,
                    algorithm_used="OSRM + KNN + Random Forest",
                    recommended_route=best_route["route_id"],
                    distance_km=best_route["distance_km"],
                    travel_time_min=best_route["predicted_time_min"],
                    congestion_level=best_route["congestion_level"]
                )

    # Render results if present in session
    if st.session_state.get("route_computed"):
        src_lat, src_lon, src_name = st.session_state["src_info"]
        dst_lat, dst_lon, dst_name = st.session_state["dst_info"]
        eval_result = st.session_state["eval_result"]
        routes = eval_result["routes"]
        rec_id = eval_result["recommended_route_id"]

        st.markdown("---")
        # Recommendation Banner
        st.success(eval_result["recommendation_explanation"])

        # Layout: Left column Map, Right column Comparison & ML Insights
        map_col, info_col = st.columns([1.2, 1.0])

        with map_col:
            st.subheader("🗺️ Interactive Route Map (OpenStreetMap & Folium)")
            center_lat = (src_lat + dst_lat) / 2.0
            center_lon = (src_lon + dst_lon) / 2.0

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

            # Source Marker
            folium.Marker(
                [src_lat, src_lon],
                popup=f"<b>Source:</b> {src_name}",
                tooltip="Source",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)

            # Destination Marker
            folium.Marker(
                [dst_lat, dst_lon],
                popup=f"<b>Destination:</b> {dst_name}",
                tooltip="Destination",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)

            # Color scheme for routes
            colors = {"Route A": "#16a34a", "Route B": "#2563eb", "Route C": "#9333ea"}

            for r in routes:
                r_id = r["route_id"]
                is_rec = (r_id == rec_id)
                route_color = colors.get(r_id, "#4b5563")
                weight = 6 if is_rec else 4
                opacity = 0.9 if is_rec else 0.65

                popup_html = f"""
                <div style='font-family: sans-serif; font-size: 13px;'>
                    <b>{r_id} ({r['route_name']})</b><br>
                    Distance: {r['distance_km']} km<br>
                    Travel Time: {r['predicted_time_min']} min<br>
                    Congestion: <b>{r['congestion_level']}</b><br>
                    Status: {'<b>⭐ RECOMMENDED</b>' if is_rec else 'Alternative'}
                </div>
                """
                folium.PolyLine(
                    locations=r["coordinates"],
                    color=route_color,
                    weight=weight,
                    opacity=opacity,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{r_id} ({r['distance_km']} km, {r['predicted_time_min']} min)"
                ).add_to(m)

            st_folium(m, width="100%", height=480)

        with info_col:
            st.subheader("📊 Comparative Route Analysis")

            for r in routes:
                is_rec = (r["route_id"] == rec_id)
                badge_class = f"badge-{r['congestion_level'].lower()}"

                with st.expander(f"{'⭐ [RECOMMENDED] ' if is_rec else ''}{r['route_id']} – {r['route_name']}", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("📏 Distance", f"{r['distance_km']} km")
                        st.markdown(f"**Congestion (KNN):** <span class='{badge_class}'>{r['congestion_level']}</span>", unsafe_allow_html=True)
                    with c2:
                        st.metric("⏱️ Est. Travel Time (RF)", f"{r['predicted_time_min']} min")
                        st.metric("🚗 Est. Speed", f"{r['estimated_speed_kmh']} km/h")

                    st.caption(f"**Cluster Pattern (K-Means):** {r['traffic_cluster']}")
                    st.caption(f"**Engine:** {r['source_type']} | **Composite Decision Score:** `{r['composite_score']}`")


# ==========================================
# TAB 2: Dijkstra vs A* Graph Analysis
# ==========================================
with tab_algorithms:
    st.subheader("⚡ Shortest-Distance Graph Theory Comparison")
    st.markdown(
        "Demonstrates graph routing on the major Pune arterial road network. "
        "Compares **Dijkstra's Algorithm** (uninformed greedy graph traversal) against "
        "**A* Search Algorithm** (informed heuristic search using Haversine straight-line distance)."
    )

    g_col1, g_col2 = st.columns(2)
    pune_nodes = sorted(list(PUNE_ROAD_NETWORK.keys()))
    with g_col1:
        g_start = st.selectbox("Select Graph Origin", options=pune_nodes, index=pune_nodes.index("Hinjewadi"))
    with g_col2:
        g_end = st.selectbox("Select Graph Destination", options=pune_nodes, index=pune_nodes.index("Hadapsar"))

    if st.button("Run Graph Algorithm Benchmark", use_container_width=True):
        if g_start == g_end:
            st.warning("Please choose distinct origin and destination nodes.")
        else:
            comp = compare_algorithms(g_start, g_end)
            d_res = comp["dijkstra"]
            a_res = comp["astar"]

            c_dijkstra, c_astar = st.columns(2)

            with c_dijkstra:
                st.markdown("### 🔵 Dijkstra's Algorithm")
                if d_res["success"]:
                    st.metric("Total Path Distance", f"{d_res['total_distance_km']} km")
                    st.metric("Nodes Visited", d_res["nodes_visited_count"])
                    st.metric("Execution Time", f"{d_res['execution_time_ms']} ms")
                    st.markdown("**Path Sequence:**")
                    st.write(" ➔ ".join(d_res["path"]))
                    with st.expander("Node Expansion Order"):
                        st.write(" ➔ ".join(d_res["visited_order"]))
                else:
                    st.error(d_res.get("error"))

            with c_astar:
                st.markdown("### 🟢 A* Search Algorithm")
                if a_res["success"]:
                    st.metric("Total Path Distance", f"{a_res['total_distance_km']} km")
                    st.metric("Nodes Visited", a_res["nodes_visited_count"])
                    st.metric("Execution Time", f"{a_res['execution_time_ms']} ms")
                    st.markdown("**Path Sequence:**")
                    st.write(" ➔ ".join(a_res["path"]))
                    with st.expander("Node Expansion Order"):
                        st.write(" ➔ ".join(a_res["visited_order"]))
                else:
                    st.error(a_res.get("error"))

            st.markdown("---")
            saved = comp["nodes_saved_by_astar"]
            if saved > 0:
                st.success(
                    f"🎯 **Academic Observation:** A* Search explored **{saved} fewer nodes** than Dijkstra "
                    f"to reach the same optimal distance ({a_res['total_distance_km']} km), "
                    f"because its Haversine heuristic function *h(n)* effectively guides search toward the destination."
                )
            else:
                st.info("Both algorithms explored a similar number of nodes on this specific sub-path.")


# ==========================================
# TAB 3: Search History (PostgreSQL)
# ==========================================
with tab_history:
    st.subheader("📜 Route Search History")
    st.markdown(f"Persistent logging of user queries via **{db.backend}**.")

    h_col1, h_col2 = st.columns([3, 1])
    with h_col2:
        num_records = st.selectbox("Display Limit", [5, 10, 20, 50], index=1)
        if st.button("🔄 Refresh History"):
            st.rerun()

    history_records = db.get_recent_history(limit=num_records)

    if history_records:
        df_history = pd.DataFrame(history_records)
        df_history.columns = [
            "Timestamp", "Source", "Destination", "Algorithm / Engine",
            "Recommended Route", "Distance (km)", "Travel Time (min)", "Congestion Level"
        ]
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("No search transactions logged yet. Perform a route search in Tab 1 to record history.")

    if "Fallback" in db.backend:
        st.markdown("---")
        with st.expander("🔧 How to connect your local PostgreSQL database"):
            st.markdown("""
            By default, RouteWise is configured to connect to PostgreSQL at:
            - **Host:** `localhost:5432`
            - **Database:** `routewise`
            - **User:** `postgres`
            - **Password:** `postgres`

            **Steps to activate PostgreSQL on your machine:**
            1. Open pgAdmin or psql shell: `CREATE DATABASE routewise;`
            2. Verify PostgreSQL service is running on port 5432.
            3. Set environment variables (optional):
               ```powershell
               $env:PG_HOST="localhost"
               $env:PG_DATABASE="routewise"
               $env:PG_USER="postgres"
               $env:PG_PASSWORD="your_password"
               ```
            4. Restart the Streamlit application. It will automatically detect PostgreSQL!
            """)


# ==========================================
# TAB 4: Academic Documentation & Viva Prep
# ==========================================
with tab_academic:
    st.subheader("🎓 Academic Documentation & Viva Defense Guide")
    st.markdown("Comprehensive reference designed for M.Tech First Year project viva and evaluation.")

    viva_section = st.radio(
        "Select Documentation Section:",
        ["1. Problem Statement & Architecture", "2. Working of All 5 Algorithms", "3. Dataset & Preprocessing", "4. Model Evaluation & Results", "5. Viva Questions & Answers"],
        horizontal=True
    )

    if "1. Problem Statement" in viva_section:
        st.markdown("""
        ### Problem Statement
        In modern metropolitan cities like Pune, standard navigation applications often focus solely on static shortest distance, 
        frequently routing commuters into severe bottlenecks during peak hours (e.g. Hinjewadi flyover, Swargate, or Senapati Bapat Road). 
        Commuters need an intelligent, multi-criteria recommendation system that evaluates alternative corridors, predicts dynamic congestion 
        using Machine Learning, and recommends paths balancing distance, time, and traffic severity.

        ### System Architecture
        ```
        [User Input: Pune Source & Destination]
                         │
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │                 Streamlit Web Frontend                 │
        └────────────────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        ┌──────────────────┐           ┌──────────────────┐
        │  Routing Engine  │           │ Graph Theory Lab │
        │  (OSRM + Folium) │           │ (Dijkstra vs A*) │
        └────────┬─────────┘           └──────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────────────────────────┐
        │               Machine Learning Pipeline                │
        │   • KNN: Classify Traffic Congestion (Low/Med/High)     │
        │   • Random Forest: Predict Travel Time in Minutes      │
        │   • K-Means: Congestion Pattern Clustering             │
        └────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────────┐
        │         Multi-Criteria Recommendation Engine           │
        │      Score = 0.50*Time + 0.25*Dist + 0.25*Congestion   │
        └────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────────┐
        │      PostgreSQL Database (Search History Logging)      │
        └────────────────────────────────────────────────────────┘
        ```
        """)

    elif "2. Working of All 5 Algorithms" in viva_section:
        st.markdown("""
        ### 1. Dijkstra's Algorithm
        - **Purpose:** Finds the guaranteed shortest-distance path on a weighted graph.
        - **Formula:** Continually relaxes edges using:  
          $$d(v) = \\min(d(v), d(u) + w(u, v))$$
        - **Where used:** In the Graph Theory Analysis tab to evaluate the baseline shortest route on Pune's major road network.
        - **Characteristics:** Uninformed search; explores all directions equally without goal-direction.

        ### 2. A* Search Algorithm
        - **Purpose:** Finds the optimal path faster than Dijkstra by using a heuristic function.
        - **Formula:**  
          $$f(n) = g(n) + h(n)$$
          where $g(n)$ is actual cost from start, and $h(n)$ is the Haversine direct straight-line distance to goal.
        - **Where used:** In the Graph Analysis tab as an efficient alternative to Dijkstra.
        - **Characteristics:** Admissible ($h(n) \\le h^*(n)$) and consistent, ensuring fewer expanded nodes.

        ### 3. K-Nearest Neighbors (KNN)
        - **Purpose:** Classifies traffic congestion level into **Low**, **Medium**, or **High**.
        - **Features:** Distance, Hour of Day, Day of Week, Weekend Flag, Estimated Speed.
        - **Metric:** Euclidean distance in scaled feature space with $k=5$.
        - **Where used:** Predicts traffic intensity for each candidate route.

        ### 4. Random Forest Regressor
        - **Purpose:** Predicts estimated travel time in minutes.
        - **Mechanism:** Ensemble of 100 decision trees trained using Bootstrap Aggregation (bagging) with feature randomness.
        - **Features:** Distance, Hour, Day, Weekend, Encoded Congestion Level, Estimated Speed.
        - **Where used:** Supplies dynamic travel time estimation for route comparison.

        ### 5. K-Means Clustering
        - **Purpose:** Unsupervised grouping of traffic observations into 3 distinct congestion regimes.
        - **Objective:** Minimizes within-cluster sum of squares (WCSS / Inertia):
          $$J = \\sum_{i=1}^{k} \\sum_{x \\in S_i} ||x - \\mu_i||^2$$
        - **Clusters Identified:**
          - **Cluster 0:** Moderate Urban Transit (Speed ~29 km/h)
          - **Cluster 1:** Severe Peak-Hour Congestion (Speed ~16 km/h)
          - **Cluster 2:** Off-Peak Free Flow (Speed ~36 km/h)
        """)

    elif "3. Dataset & Preprocessing" in viva_section:
        st.markdown("""
        ### Dataset Details
        - **Name:** `traffic_data.csv` (1,600 observations)
        - **Scope:** Major Pune arterial corridors (Shivajinagar, Kothrud, Hinjewadi, Viman Nagar, Hadapsar, Swargate, etc.).
        - **Labeling:** Clearly labelled as simulated dataset calibrated against typical Pune rush-hour profiles.
        - **Preprocessing steps:**
          1. Feature engineering: extraction of `is_weekend`, peak-hour indicator, and signal delay factors.
          2. Label Encoding: Target congestion categories (`High: 0`, `Medium: 1`, `Low: 2`).
          3. Standardization: `StandardScaler` applied to continuous variables for KNN and K-Means.
          4. Train-Test Split: 80% Training (1,280 samples) and 20% Testing (320 samples) with stratification.
        """)

    elif "4. Model Evaluation & Results" in viva_section:
        st.markdown("""
        ### Model Performance Metrics
        | Model | Task | Metric | Performance |
        |---|---|---|---|
        | **K-Nearest Neighbors (KNN)** | Classification | Accuracy | **90.94%** (Macro F1: 0.91) |
        | **Random Forest Regressor** | Regression | Mean Absolute Error (MAE) | **1.83 minutes** |
        | **Random Forest Regressor** | Regression | $R^2$ Score | **0.9937** |
        | **K-Means Clustering** | Unsupervised | Silhouette Score ($k=3$) | **0.3361** |
        | **A* vs Dijkstra** | Graph Search | Node Expansion Reduction | **~15% to 25% fewer nodes** |
        """)

    elif "5. Viva Questions & Answers" in viva_section:
        st.markdown("""
        ### Top Viva Defense Questions & Academic Answers

        **Q1: Why did you choose A* over Dijkstra for route navigation?**  
        *Answer:* While Dijkstra explores nodes in all directions uniformly, A* incorporates a heuristic estimate ($h(n)$) representing straight-line Haversine distance to the goal. Because Haversine distance never overestimates actual road distance (it is admissible and consistent), A* is guaranteed to return the optimal shortest path while visiting significantly fewer search states.

        **Q2: Why use Random Forest instead of a simple Linear Regression for travel time?**  
        *Answer:* Urban traffic travel time is non-linear due to rush-hour inflection points, traffic signals, and bottle-necking. Random Forest handles non-linear feature interactions and threshold effects without requiring manual polynomial transformations.

        **Q3: How does your recommendation system balance distance versus travel time?**  
        *Answer:* We use a Multi-Criteria Decision Analysis (MCDA) scoring function:  
        $$\\text{Score} = 0.50 \\times \\text{NormTime} + 0.25 \\times \\text{NormDist} + 0.25 \\times \\text{CongestionPenalty}$$  
        A route that is 1 km longer but avoids a 15-minute gridlock will obtain a superior composite score and be recommended to the commuter.

        **Q4: How is PostgreSQL integrated and what happens if the database server is offline?**  
        *Answer:* We implemented an enterprise design pattern in `database.py` using `psycopg2`. It connects to PostgreSQL to record route transactions. If PostgreSQL is offline, our resilient wrapper automatically switches to an internal SQLite fallback so the application maintains continuous availability without throwing uncaught exceptions.
        """)
