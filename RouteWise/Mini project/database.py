"""
RouteWise - Database Management Module (PostgreSQL with resilient SQLite Fallback)
Academic Level: M.Tech First Year (Mini-Project)
Team: 3 Members | City: Pune, Maharashtra

This module provides:
1. PostgreSQL connection via psycopg2 for persistent route search history.
2. Table creation: `route_history`.
3. Inserting search and recommendation transactions.
4. Fetching recent search history.
5. Resilient fallback to SQLite when PostgreSQL server is not locally active or configured,
   ensuring the application never crashes during academic evaluation.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# PostgreSQL Default Configuration (Standard academic setup)
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE", "routewise")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

SQLITE_FALLBACK_FILE = "routewise_history.db"


class DatabaseManager:
    """
    Manages connection to PostgreSQL, with graceful fallback to SQLite
    to guarantee high reliability during project presentation.
    """
    def __init__(self, host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.backend = "unknown"
        self._init_connection()

    def _init_connection(self):
        """Attempts PostgreSQL connection; falls back to SQLite if PG is unavailable."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                connect_timeout=2
            )
            conn.close()
            self.backend = "PostgreSQL"
            self._init_postgres_schema()
        except Exception:
            # Fallback to local SQLite file
            self.backend = "SQLite (PostgreSQL Fallback)"
            self._init_sqlite_schema()

    def _init_postgres_schema(self):
        """Creates the route_history table in PostgreSQL if not present."""
        import psycopg2
        query = """
        CREATE TABLE IF NOT EXISTS route_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            source VARCHAR(150) NOT NULL,
            destination VARCHAR(150) NOT NULL,
            algorithm_used VARCHAR(50),
            recommended_route VARCHAR(100),
            distance_km REAL,
            travel_time_min REAL,
            congestion_level VARCHAR(20)
        );
        """
        try:
            with psycopg2.connect(
                host=self.host, port=self.port, dbname=self.dbname,
                user=self.user, password=self.password
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                conn.commit()
        except Exception as e:
            print(f"Warning: PostgreSQL schema init failed: {e}")
            self.backend = "SQLite (PostgreSQL Fallback)"
            self._init_sqlite_schema()

    def _init_sqlite_schema(self):
        """Creates the route_history table in SQLite."""
        query = """
        CREATE TABLE IF NOT EXISTS route_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT NOT NULL,
            destination TEXT NOT NULL,
            algorithm_used TEXT,
            recommended_route TEXT,
            distance_km REAL,
            travel_time_min REAL,
            congestion_level TEXT
        );
        """
        with sqlite3.connect(SQLITE_FALLBACK_FILE) as conn:
            conn.execute(query)
            conn.commit()

    def save_search(self, source: str, destination: str, algorithm_used: str,
                    recommended_route: str, distance_km: float, travel_time_min: float,
                    congestion_level: str) -> bool:
        """
        Records a search and recommendation event into the database.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "PostgreSQL" in self.backend and "Fallback" not in self.backend:
            try:
                import psycopg2
                query = """
                INSERT INTO route_history 
                (source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                with psycopg2.connect(
                    host=self.host, port=self.port, dbname=self.dbname,
                    user=self.user, password=self.password
                ) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, (source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level))
                    conn.commit()
                return True
            except Exception:
                pass

        # SQLite fallback execution
        try:
            query = """
            INSERT INTO route_history 
            (timestamp, source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            with sqlite3.connect(SQLITE_FALLBACK_FILE) as conn:
                conn.execute(query, (now_str, source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            return False

    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """
        Fetches recent search transactions.
        """
        if "PostgreSQL" in self.backend and "Fallback" not in self.backend:
            try:
                import psycopg2
                query = """
                SELECT timestamp, source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level
                FROM route_history
                ORDER BY id DESC
                LIMIT %s;
                """
                with psycopg2.connect(
                    host=self.host, port=self.port, dbname=self.dbname,
                    user=self.user, password=self.password
                ) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, (limit,))
                        rows = cur.fetchall()
                        return [
                            {
                                "timestamp": str(r[0]),
                                "source": r[1],
                                "destination": r[2],
                                "algorithm": r[3],
                                "recommended_route": r[4],
                                "distance_km": round(float(r[5]), 1),
                                "travel_time_min": round(float(r[6]), 1),
                                "congestion_level": r[7]
                            }
                            for r in rows
                        ]
            except Exception:
                pass

        # SQLite fallback retrieval
        try:
            query = """
            SELECT timestamp, source, destination, algorithm_used, recommended_route, distance_km, travel_time_min, congestion_level
            FROM route_history
            ORDER BY id DESC
            LIMIT ?;
            """
            with sqlite3.connect(SQLITE_FALLBACK_FILE) as conn:
                cur = conn.cursor()
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                return [
                    {
                        "timestamp": str(r[0]),
                        "source": r[1],
                        "destination": r[2],
                        "algorithm": r[3],
                        "recommended_route": r[4],
                        "distance_km": round(float(r[5]), 1),
                        "travel_time_min": round(float(r[6]), 1),
                        "congestion_level": r[7]
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"Error reading history: {e}")
            return []


if __name__ == "__main__":
    db = DatabaseManager()
    print(f"Active Database Backend: {db.backend}")
    db.save_search("Hinjewadi", "Hadapsar", "A* Search", "Route A", 31.2, 48.5, "Medium")
    history = db.get_recent_history(5)
    print("Recent history entries:", history)
