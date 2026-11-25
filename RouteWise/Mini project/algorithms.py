"""
RouteWise - Graph Algorithms Module (Dijkstra and A* Search)
Academic Level: M.Tech First Year (Mini-Project)
Team: 3 Members | City: Pune, Maharashtra

This module provides:
1. Academic graph representation of key Pune road junctions with latitude/longitude.
2. Dijkstra's Algorithm (Greedy shortest path based on edge weights/distance).
3. A* Search Algorithm (Informed search with Haversine distance heuristic).
4. Algorithm performance comparison (path length, execution time, nodes visited).
"""

import math
import heapq
import time
from typing import Dict, List, Tuple, Optional

# Pune major arterial road network graph with real coordinates (latitude, longitude)
PUNE_ROAD_NETWORK: Dict[str, Tuple[float, float]] = {
    "Swargate": (18.5018, 73.8586),
    "Deccan Gymkhana": (18.5173, 73.8415),
    "Shivajinagar": (18.5314, 73.8446),
    "Kothrud": (18.5074, 73.8077),
    "Baner": (18.5590, 73.7868),
    "Aundh": (18.5602, 73.8031),
    "Hinjewadi": (18.5913, 73.7389),
    "Wakad": (18.5987, 73.7661),
    "Pimpri": (18.6279, 73.7997),
    "Viman Nagar": (18.5679, 73.9143),
    "Kalyani Nagar": (18.5463, 73.9033),
    "Koregaon Park": (18.5362, 73.8940),
    "Hadapsar": (18.5089, 73.9259),
    "Magarpatta": (18.5158, 73.9272),
    "Katraj": (18.4575, 73.8677),
}

# Road connections between Pune nodes with realistic road distances (in km)
PUNE_EDGES = [
    # Core Central Connections
    ("Swargate", "Deccan Gymkhana", 3.2),
    ("Swargate", "Katraj", 6.5),
    ("Swargate", "Hadapsar", 8.4),
    ("Swargate", "Shivajinagar", 4.5),
    ("Deccan Gymkhana", "Shivajinagar", 2.3),
    ("Deccan Gymkhana", "Kothrud", 4.1),
    
    # West Corridor (Kothrud, Baner, Hinjewadi)
    ("Kothrud", "Baner", 7.8),
    ("Kothrud", "Deccan Gymkhana", 4.1),
    ("Baner", "Aundh", 3.5),
    ("Baner", "Wakad", 5.2),
    ("Baner", "Hinjewadi", 8.6),
    ("Wakad", "Hinjewadi", 4.8),
    ("Wakad", "Pimpri", 6.1),
    ("Aundh", "Shivajinagar", 6.2),
    ("Aundh", "Pimpri", 7.5),
    
    # East Corridor (Shivajinagar, Koregaon Park, Viman Nagar, Hadapsar)
    ("Shivajinagar", "Koregaon Park", 5.6),
    ("Shivajinagar", "Viman Nagar", 9.8),
    ("Koregaon Park", "Kalyani Nagar", 2.8),
    ("Kalyani Nagar", "Viman Nagar", 3.4),
    ("Kalyani Nagar", "Magarpatta", 6.2),
    ("Viman Nagar", "Hadapsar", 9.2),
    ("Magarpatta", "Hadapsar", 2.5),
    ("Koregaon Park", "Swargate", 6.8),
]


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Computes great-circle distance between two GPS coordinates in kilometers.
    Used as an admissible and consistent heuristic function h(n) in A* search.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth's radius in kilometers

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class PuneGraph:
    """
    Weighted road graph representation.
    """
    def __init__(self):
        self.nodes = PUNE_ROAD_NETWORK
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {node: [] for node in self.nodes}
        self._build_graph()

    def _build_graph(self):
        """Constructs bidirectional road graph."""
        for u, v, weight in PUNE_EDGES:
            if u in self.adjacency and v in self.adjacency:
                self.adjacency[u].append((v, weight))
                self.adjacency[v].append((u, weight))

    def find_nearest_node(self, lat: float, lon: float) -> str:
        """Finds the closest named node in the graph for arbitrary GPS coordinates."""
        closest_node = None
        min_dist = float("inf")
        for name, coord in self.nodes.items():
            dist = haversine_distance((lat, lon), coord)
            if dist < min_dist:
                min_dist = dist
                closest_node = name
        return closest_node


def dijkstra_algorithm(graph: PuneGraph, start_node: str, end_node: str) -> dict:
    """
    Dijkstra's Algorithm:
    - Finds the shortest-distance path in a non-negative weighted graph.
    - Uses a min-priority queue (heapq).
    - Guarantees an optimal shortest path without requiring a heuristic.
    """
    start_time = time.perf_counter()

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return {"success": False, "error": "Invalid start or end node"}

    distances = {node: float("inf") for node in graph.nodes}
    previous = {node: None for node in graph.nodes}
    distances[start_node] = 0.0

    # Priority queue stores tuples: (current_distance, node)
    pq = [(0.0, start_node)]
    nodes_visited_count = 0
    visited_order = []

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        nodes_visited_count += 1
        visited_order.append(current_node)

        if current_node == end_node:
            break

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph.adjacency[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    exec_time_ms = (time.perf_counter() - start_time) * 1000.0

    # Reconstruct path
    path = []
    curr = end_node
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()

    if not path or path[0] != start_node:
        return {"success": False, "error": "No reachable path found"}

    return {
        "success": True,
        "algorithm": "Dijkstra",
        "path": path,
        "total_distance_km": round(distances[end_node], 2),
        "nodes_visited_count": nodes_visited_count,
        "visited_order": visited_order,
        "execution_time_ms": round(exec_time_ms, 3)
    }


def astar_algorithm(graph: PuneGraph, start_node: str, end_node: str) -> dict:
    """
    A* Search Algorithm:
    - Informed search combining actual cost g(n) with heuristic estimate h(n):
        f(n) = g(n) + h(n)
    - Heuristic h(n) is the Haversine direct line-of-sight distance to the goal.
    - Since straight-line distance is always <= actual road distance,
      the heuristic is admissible and consistent, guaranteeing an optimal path
      while expanding fewer nodes than Dijkstra.
    """
    start_time = time.perf_counter()

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return {"success": False, "error": "Invalid start or end node"}

    goal_coord = graph.nodes[end_node]

    # g_score[node] is cost from start to node
    g_score = {node: float("inf") for node in graph.nodes}
    g_score[start_node] = 0.0

    # f_score[node] = g_score[node] + h(node)
    f_score = {node: float("inf") for node in graph.nodes}
    f_score[start_node] = haversine_distance(graph.nodes[start_node], goal_coord)

    previous = {node: None for node in graph.nodes}

    # Priority queue stores tuples: (f_score, g_score, node)
    pq = [(f_score[start_node], 0.0, start_node)]
    nodes_visited_count = 0
    visited_order = []

    while pq:
        current_f, current_g, current_node = heapq.heappop(pq)
        nodes_visited_count += 1
        visited_order.append(current_node)

        if current_node == end_node:
            break

        if current_g > g_score[current_node]:
            continue

        for neighbor, weight in graph.adjacency[current_node]:
            tentative_g = current_g + weight
            if tentative_g < g_score[neighbor]:
                previous[neighbor] = current_node
                g_score[neighbor] = tentative_g
                h_cost = haversine_distance(graph.nodes[neighbor], goal_coord)
                f_cost = tentative_g + h_cost
                f_score[neighbor] = f_cost
                heapq.heappush(pq, (f_cost, tentative_g, neighbor))

    exec_time_ms = (time.perf_counter() - start_time) * 1000.0

    # Reconstruct path
    path = []
    curr = end_node
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()

    if not path or path[0] != start_node:
        return {"success": False, "error": "No reachable path found"}

    return {
        "success": True,
        "algorithm": "A* Search",
        "path": path,
        "total_distance_km": round(g_score[end_node], 2),
        "nodes_visited_count": nodes_visited_count,
        "visited_order": visited_order,
        "execution_time_ms": round(exec_time_ms, 3)
    }


def compare_algorithms(start_node: str, end_node: str) -> dict:
    """
    Executes both Dijkstra and A* and produces a structured academic comparison.
    """
    graph = PuneGraph()
    dijkstra_res = dijkstra_algorithm(graph, start_node, end_node)
    astar_res = astar_algorithm(graph, start_node, end_node)

    return {
        "start": start_node,
        "end": end_node,
        "dijkstra": dijkstra_res,
        "astar": astar_res,
        "nodes_saved_by_astar": (
            dijkstra_res.get("nodes_visited_count", 0) - astar_res.get("nodes_visited_count", 0)
            if dijkstra_res.get("success") and astar_res.get("success") else 0
        )
    }


if __name__ == "__main__":
    test_start = "Hinjewadi"
    test_end = "Hadapsar"
    print(f"Comparing Dijkstra vs A* from {test_start} to {test_end}:")
    comp = compare_algorithms(test_start, test_end)
    print("Dijkstra Result:", comp["dijkstra"])
    print("A* Result:", comp["astar"])
    print(f"Nodes saved by A* heuristic guidance: {comp['nodes_saved_by_astar']}")
