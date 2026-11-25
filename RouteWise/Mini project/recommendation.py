"""
RouteWise - Route Recommendation Engine
Academic Level: M.Tech First Year (Mini-Project)
Team: 3 Members | City: Pune, Maharashtra

This module combines:
1. Distance calculation from routing engine.
2. Estimated travel time from Random Forest Regressor.
3. Congestion level from K-Nearest Neighbors (KNN).
4. Congestion cluster from K-Means.
5. Multi-Criteria Decision Analysis (MCDA) to recommend the optimal route.
"""

from typing import List, Dict
from traffic_prediction import TrafficPredictor


class RouteRecommender:
    """
    Evaluates candidate routes and recommends the optimal one
    based on predicted travel time, distance, and congestion penalty.
    """
    def __init__(self, predictor: TrafficPredictor = None):
        self.predictor = predictor or TrafficPredictor()

    def evaluate_routes(self, routes: List[Dict], hour: int, day_of_week: int) -> Dict:
        """
        Takes raw route geometries/distances and produces ML-enriched comparison metrics,
        calculating an optimal recommendation score.
        """
        if not routes:
            return {"error": "No routes available for evaluation"}

        enriched_routes = []
        congestion_penalty_map = {"Low": 1.0, "Medium": 2.2, "High": 3.8}

        for idx, route in enumerate(routes):
            dist = route["distance_km"]
            # Predict congestion with KNN
            # If alternative route, simulate slight variation in local traffic flow
            if idx == 0:
                speed_est = None  # standard corridor speed
            else:
                # Alternative route bypasses may have higher speed during peak hours
                base_speed = self.predictor.estimate_speed_by_context(hour, day_of_week)
                speed_est = base_speed * 1.12 if base_speed < 22 else base_speed * 0.95

            knn_res = self.predictor.predict_congestion(dist, hour, day_of_week, avg_speed=speed_est)
            cong_level = knn_res["congestion_level"]

            # Predict Travel Time with Random Forest
            pred_time = self.predictor.predict_travel_time(dist, hour, day_of_week, congestion_level=cong_level)

            # Analyze cluster with K-Means
            cluster_res = self.predictor.analyze_traffic_cluster(hour, knn_res["estimated_speed_kmh"], pred_time)

            enriched_routes.append({
                "route_id": route["route_id"],
                "route_name": route.get("route_name", f"Route {idx+1}"),
                "distance_km": dist,
                "osrm_duration_min": route.get("osrm_duration_min", pred_time),
                "predicted_time_min": pred_time,
                "congestion_level": cong_level,
                "congestion_probabilities": knn_res["probabilities"],
                "estimated_speed_kmh": knn_res["estimated_speed_kmh"],
                "traffic_cluster": cluster_res["description"],
                "coordinates": route["coordinates"],
                "source_type": route.get("source_type", "Standard Routing Engine")
            })

        # Multi-Criteria Scoring:
        # Score = (w_time * norm_time) + (w_dist * norm_dist) + (w_cong * norm_cong)
        # Weights: 50% Travel Time, 25% Distance, 25% Congestion Penalty
        w_time = 0.50
        w_dist = 0.25
        w_cong = 0.25

        max_time = max(r["predicted_time_min"] for r in enriched_routes) or 1.0
        max_dist = max(r["distance_km"] for r in enriched_routes) or 1.0
        max_penalty = 3.8

        best_score = float("inf")
        best_route_idx = 0

        for idx, r in enumerate(enriched_routes):
            norm_time = r["predicted_time_min"] / max_time
            norm_dist = r["distance_km"] / max_dist
            penalty = congestion_penalty_map.get(r["congestion_level"], 2.0)
            norm_cong = penalty / max_penalty

            composite_score = (w_time * norm_time) + (w_dist * norm_dist) + (w_cong * norm_cong)
            r["composite_score"] = round(composite_score, 3)

            if composite_score < best_score:
                best_score = composite_score
                best_route_idx = idx

        recommended_route = enriched_routes[best_route_idx]

        # Generate student & user-friendly explanation
        explanation = self._generate_recommendation_reason(enriched_routes, best_route_idx)

        return {
            "routes": enriched_routes,
            "recommended_route_id": recommended_route["route_id"],
            "recommended_route_name": recommended_route["route_name"],
            "recommendation_explanation": explanation
        }

    def _generate_recommendation_reason(self, routes: List[Dict], best_idx: int) -> str:
        """Constructs plain-language reasoning for the recommended choice."""
        best = routes[best_idx]
        other_routes = [r for i, r in enumerate(routes) if i != best_idx]

        if not other_routes:
            return f"{best['route_name']} is the primary path with predicted travel time of {best['predicted_time_min']} min."

        other = other_routes[0]
        time_diff = abs(round(other["predicted_time_min"] - best["predicted_time_min"], 1))
        dist_diff = abs(round(other["distance_km"] - best["distance_km"], 1))

        if best["predicted_time_min"] < other["predicted_time_min"] and best["congestion_level"] != "High":
            return (
                f"🌟 **{best['route_id']} ({best['route_name']})** is recommended: "
                f"It saves ~**{time_diff} minutes** over the alternative and encounters "
                f"**{best['congestion_level']} traffic congestion**, providing a faster and smoother drive."
            )
        elif best["distance_km"] <= other["distance_km"]:
            return (
                f"🌟 **{best['route_id']} ({best['route_name']})** is recommended: "
                f"It is the most direct trajectory (**{best['distance_km']} km**) with an estimated travel time "
                f"of **{best['predicted_time_min']} minutes** under {best['congestion_level']} congestion."
            )
        else:
            return (
                f"🌟 **{best['route_id']} ({best['route_name']})** is recommended based on overall "
                f"balanced trade-off between transit distance ({best['distance_km']} km) and "
                f"predicted traffic delay ({best['predicted_time_min']} min)."
            )
