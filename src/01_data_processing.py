#!/usr/bin/env python
import pandas as pd
from pathlib import Path


DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/app/output")



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Bemeneti adatok beolvasása
    veh_path = DATA_DIR / "vehicle_updates.csv"
    stop_times_path = DATA_DIR / "stop_times.txt"
    trips_path = DATA_DIR / "trips.txt"

    df = pd.read_csv(veh_path)
    stop_times = pd.read_csv(stop_times_path)
    trips = pd.read_csv(trips_path)  

    # 2) Oszlopok egységesítése (notebook szerint)
    # delay_seconds_calc -> delay_seconds; stop_id -> last_stop_id
    df = df.rename(
        columns={
            "delay_seconds_calc": "delay_seconds",
            "stop_id": "last_stop_id",
        }
    )

    # 3) Végállomás (end_stop_id) számítása a stop_times-ból
    end_stops = (
        stop_times
        .sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")
        .last()
        .reset_index()[["trip_id", "stop_id"]]
        .rename(columns={"stop_id": "end_stop_id"})
    )

    # 4) Merge: jármű frissítések + végállomás
    veh = df.merge(end_stops, on="trip_id", how="left")

    # 5) run_id definiálása (trip_id + vehicle_id változás alapján)
    veh = veh.sort_values(["vehicle_id", "vehicle_timestamp"])
    veh["run_id"] = (
        (veh["trip_id"] != veh["trip_id"].shift()) |
        (veh["vehicle_id"] != veh["vehicle_id"].shift())
    ).cumsum()

    # 6) Végkésés (y_end_delay) becsatolása:
    # csak azok a sorok, ahol last_stop_id == end_stop_id -> csoportosítás run_id szerint
    end_delay = (
        veh[veh["last_stop_id"] == veh["end_stop_id"]]
        .groupby("run_id")["delay_seconds"]
        .first()
        .reset_index()
        .rename(columns={"delay_seconds": "y_end_delay"})
    )

    veh = veh.merge(end_delay, on="run_id", how="left")
    veh = veh.dropna(subset=["y_end_delay"])

    # 7) Duplikált oszlopnevek tisztítása – utolsó előfordulás marad
    veh = veh.loc[:, ~veh.columns.duplicated(keep="last")]

    # 8) Végső oszlopnevek egységesítése (_calc)
    veh = veh.rename(
        columns={
            "delay_seconds": "delay_seconds_calc",
            "y_end_delay": "y_end_delay_calc",
        }
    )

    # 9) Dátum/idő konverzió és idő-alapú feature-ök
    for col in ["timestamp_utc", "vehicle_timestamp", "scheduled_arrival"]:
        if col in veh.columns:
            veh[col] = pd.to_datetime(veh[col])

    veh["hour"] = veh["vehicle_timestamp"].dt.hour
    veh["weekday"] = veh["vehicle_timestamp"].dt.weekday

    # 10) Modellhez szükséges oszlopok kiválasztása
    cols = [
        "timestamp_utc",
        "vehicle_timestamp",
        "vehicle_id",
        "trip_id",
        "route_id",
        "last_stop_id",
        "end_stop_id",
        "current_stop_sequence",
        "hour",
        "weekday",
        "delay_seconds_calc",
        "y_end_delay_calc",
    ]

    missing = [c for c in cols if c not in veh.columns]
    if missing:
        raise ValueError(f"Hiányzó elvárt oszlop(ok) a feldolgozott adatban: {missing}")

    train_df = veh[cols].dropna()

    out_path = DATA_DIR / "processed_dataset.csv"
    train_df.to_csv(out_path, index=False)
    print(f"[01_data_processing] Saved processed dataset to {out_path} ({len(train_df)} sor)")


if __name__ == "__main__":
    main()
