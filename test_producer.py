#!/usr/bin/env python3
import json
import io
import time
import random
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import os

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]


def generate_batch(n_rows: int = 100, inject_anomalies: bool = True) -> pd.DataFrame:
    base_time = datetime.now(UTC) # fixing the deprecation warning

    data = {
        "timestamp": [
            (base_time + timedelta(minutes=i)).isoformat() for i in range(n_rows)
        ],
        "temperature": np.random.normal(loc=22.0, scale=1.5, size=n_rows).round(2),
        "humidity":    np.random.normal(loc=55.0, scale=5.0, size=n_rows).round(2),
        "pressure":    np.random.normal(loc=1013.0, scale=3.0, size=n_rows).round(2),
        "wind_speed":  np.abs(np.random.normal(loc=10.0, scale=2.5, size=n_rows)).round(2),
    }

    df = pd.DataFrame(data)

    # Inject a few obvious anomalies so students can see the detector catch them
    if inject_anomalies and n_rows > 10:
        anomaly_indices = random.sample(range(n_rows), k=max(1, n_rows // 20))
        for idx in anomaly_indices:
            col = random.choice(["temperature", "humidity", "pressure", "wind_speed"])
            # Push the value 5-8 standard deviations out
            direction = random.choice([-1, 1])
            df.at[idx, col] = df[col].mean() + direction * df[col].std() * random.uniform(5, 8)

    return df


def upload_batch(df: pd.DataFrame):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") # fixing the deprecation warning
    key = f"raw/sensors_{timestamp}.csv"

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )
    print(f"Uploaded {len(df)} rows → s3://{BUCKET_NAME}/{key}")
    return key


if __name__ == "__main__":
    interval = int(os.getenv("INTERVAL_SECONDS", "60"))
    print(f"Producing batches every {interval}s. Ctrl+C to stop.")

    while True:
        df = generate_batch(n_rows=100, inject_anomalies=True)
        upload_batch(df)
        time.sleep(interval)