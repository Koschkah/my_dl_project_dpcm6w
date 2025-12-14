import os
import time
import requests
import pandas as pd
from datetime import datetime
from google.transit import gtfs_realtime_pb2
from pathlib import Path
from dotenv import load_dotenv
import math

# --- .env betöltése ---
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("BKK_API_KEY")
if not API_KEY:
    raise RuntimeError("API key not found in .env file (BKK_API_KEY=...)")

# --- Adatmappa ---
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / "vehicle_updates.csv"

# --- API URL-ek ---
URL_POS = f"https://go.bkk.hu/api/query/v1/ws/gtfs-rt/full/VehiclePositions.pb?key={API_KEY}"
URL_TRIP = f"https://go.bkk.hu/api/query/v1/ws/gtfs-rt/full/TripUpdates.pb?key={API_KEY}"

print("=== BKK FUTÁR adatgyűjtés – érkezési és indulási késés ===")
print(f"VehiclePositions: {URL_POS}")
print(f"TripUpdates: {URL_TRIP}")
print(f"Mentés: {DATA_PATH}")
print("Csak valós késéseket írunk ki (delay ≠ 0 és nem NaN).\n")

for _ in range(2):  # teszteléshez csak 2 ciklus
    try:
        # --- 1️⃣ Pozíciók lekérése ---
        res_pos = requests.get(URL_POS, headers={"Accept": "application/x-protobuf"}, timeout=10)
        res_pos.raise_for_status()
        feed_pos = gtfs_realtime_pb2.FeedMessage()
        feed_pos.ParseFromString(res_pos.content)

        # --- 2️⃣ TripUpdates lekérése (késésadatok) ---
        res_trip = requests.get(URL_TRIP, headers={"Accept": "application/x-protobuf"}, timeout=10)
        res_trip.raise_for_status()
        feed_trip = gtfs_realtime_pb2.FeedMessage()
        feed_trip.ParseFromString(res_trip.content)

        # --- Késés dictionary: trip_id -> (arrival_delay, departure_delay) ---
        delay_map = {}
        for ent in feed_trip.entity:
            if hasattr(ent, "trip_update") and hasattr(ent.trip_update.trip, "trip_id"):
                trip_id = ent.trip_update.trip.trip_id
                if len(ent.trip_update.stop_time_update) > 0:
                    stu = ent.trip_update.stop_time_update[0]
                    arrival_delay = getattr(stu.arrival, "delay", None) if stu.HasField("arrival") else None
                    departure_delay = getattr(stu.departure, "delay", None) if stu.HasField("departure") else None
                    delay_map[trip_id] = (arrival_delay, departure_delay)

        now = datetime.utcnow()
        records = []

        for entity in feed_pos.entity:
            v = entity.vehicle
            trip_id = getattr(v.trip, "trip_id", None)
            arrival_delay, departure_delay = delay_map.get(trip_id, (None, None))

            # csak akkor vesszük fel, ha van valamelyik delay
            if not all(x in [None, 0] for x in [arrival_delay, departure_delay]):
                records.append({
                    "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "vehicle_id": entity.id,
                    "trip_id": trip_id,
                    "route_id": getattr(v.trip, "route_id", None),
                    "lat": getattr(v.position, "latitude", None),
                    "lon": getattr(v.position, "longitude", None),
                    "speed": getattr(v.position, "speed", None),
                    "heading": getattr(v.position, "bearing", None),
                    "vehicle_timestamp": datetime.fromtimestamp(v.timestamp),
                    "last_stop_id": getattr(v, "stop_id", None),
                    "arrival_delay_seconds": arrival_delay,
                    "departure_delay_seconds": departure_delay
                })

        if not records:
            print(f"{now.strftime('%H:%M:%S')} – Nincs érvényes delay adat ebben a ciklusban.\n")
            continue

        df = pd.DataFrame(records)
        print(f"{now.strftime('%H:%M:%S')} – {len(df)} rekord késésadattal:")
        print(df[["trip_id", "last_stop_id", "arrival_delay_seconds", "departure_delay_seconds"]].head(10))
        df.to_csv(DATA_PATH, mode="a", header=not DATA_PATH.exists(), index=False)
        print()

    except Exception as e:
        print("Hiba:", e)

    time.sleep(5)
