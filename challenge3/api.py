"""Astar Island API client."""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.ainm.no"
TOKEN = os.getenv("AINM_TOKEN", "")
_last_request_time = 0
MIN_REQUEST_INTERVAL = 1.5  # seconds between requests


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get(path, **kwargs):
    _rate_limit()
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path, json_data=None, **kwargs):
    _rate_limit()
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=json_data, **kwargs)
    r.raise_for_status()
    return r.json()


# --- Endpoints ---

def get_rounds():
    return _get("/astar-island/rounds")


def get_round(round_id: str):
    return _get(f"/astar-island/rounds/{round_id}")


def get_budget():
    return _get("/astar-island/budget")


def simulate(round_id: str, seed_index: int, x: int, y: int, w: int = 15, h: int = 15):
    return _post("/astar-island/simulate", {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": x,
        "viewport_y": y,
        "viewport_w": w,
        "viewport_h": h,
    })


def submit(round_id: str, seed_index: int, prediction: list):
    return _post("/astar-island/submit", {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    })


def get_my_rounds():
    return _get("/astar-island/my-rounds")


def get_my_predictions(round_id: str):
    return _get(f"/astar-island/my-predictions/{round_id}")


def get_analysis(round_id: str, seed_index: int):
    return _get(f"/astar-island/analysis/{round_id}/{seed_index}")


def get_leaderboard():
    return _get("/astar-island/leaderboard")
