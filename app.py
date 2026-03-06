# app.py
import io
import json
import os
import logging
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file

# logging configuration
LOG_FILE = '/opt/anomaly-detection/app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# application setup
app = FastAPI(title="Anomaly Detection Pipeline")

try:
    s3 = boto3.client('s3')
    BUCKET_NAME = os.environ["BUCKET_NAME"]
    logger.info(f"Application started successfully. Bucket: {BUCKET_NAME}")
except KeyError as e:
    logger.error(f"BUCKET_NAME environment variable not set: {e}")
    raise
except Exception as e:
    logger.error(f"Application startup failed: {e}")
    raise

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")
        logger.info(f"Received SNS message type: {msg_type}")

        # SNS sends a SubscriptionConfirmation before it will deliver any messages.
        # Visiting the SubscribeURL confirms the subscription.
        if msg_type == "SubscriptionConfirmation":
            try:
                confirm_url = body["SubscribeURL"]
                logger.info(f"Confirming SNS subscription: {confirm_url}")
                response = requests.get(confirm_url, timeout=10)
                response.raise_for_status()
                logger.info("SNS subscription confirmed successfully")
                return {"status": "confirmed"}
            except requests.RequestException as e:
                logger.error(f"Failed to confirm SNS subscription: {e}")
                return {"status": "error", "message": str(e)}

        if msg_type == "Notification":
            try:
                # The SNS message body contains the S3 event as a JSON string
                s3_event = json.loads(body["Message"])
                logger.info(f"Processing S3 event notification")
                
                for record in s3_event.get("Records", []):
                    key = record["s3"]["object"]["key"]
                    logger.info(f"New file detected: {key}")
                    
                    if key.startswith("raw/") and key.endswith(".csv"):
                        logger.info(f"Queueing file for processing: {key}")
                        background_tasks.add_task(process_file, BUCKET_NAME, key)
                    else:
                        logger.warning(f"Skipping file (not in raw/ or not CSV): {key}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse S3 event JSON: {e}")
                return {"status": "error", "message": "Invalid JSON in SNS message"}
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
                return {"status": "error", "message": str(e)}

        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Unexpected error in /notify endpoint: {e}")
        return {"status": "error", "message": str(e)}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    try:
        logger.info(f"Fetching recent anomalies (limit={limit})")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        logger.info(f"Found {len(keys)} recent processed files")

        all_anomalies = []
        for key in keys:
            try:
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                if "anomaly" in df.columns:
                    flagged = df[df["anomaly"] == True].copy()
                    flagged["source_file"] = key
                    all_anomalies.append(flagged)
            except Exception as e:
                logger.error(f"Error reading file {key}: {e}")
                continue

        if not all_anomalies:
            logger.info("No anomalies found")
            return {"count": 0, "anomalies": []}

        combined = pd.concat(all_anomalies).head(limit)
        logger.info(f"Returning {len(combined)} anomalies")
        return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}
    
    except Exception as e:
        logger.error(f"Error in /anomalies/recent endpoint: {e}")
        return {"error": str(e)}


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try:
        logger.info("Fetching anomaly summary")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    try:
                        response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        summaries.append(json.loads(response["Body"].read()))
                    except Exception as e:
                        logger.error(f"Error reading summary {obj['Key']}: {e}")
                        continue

        if not summaries:
            logger.info("No processed files found")
            return {"message": "No processed files yet."}

        total_rows = sum(s["total_rows"] for s in summaries)
        total_anomalies = sum(s["anomaly_count"] for s in summaries)

        logger.info(f"Summary: {len(summaries)} files, {total_rows} rows, {total_anomalies} anomalies")

        return {
            "files_processed": len(summaries),
            "total_rows_scored": total_rows,
            "total_anomalies": total_anomalies,
            "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
        }
    
    except Exception as e:
        logger.error(f"Error in /anomalies/summary endpoint: {e}")
        return {"error": str(e)}


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    try:
        logger.info("Fetching current baseline")
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()

        channels = {}
        for channel, stats in baseline.items():
            if channel == "last_updated":
                continue
            channels[channel] = {
                "observations": stats["count"],
                "mean": round(stats["mean"], 4),
                "std": round(stats.get("std", 0.0), 4),
                "baseline_mature": stats["count"] >= 30,
            }

        logger.info(f"Baseline contains {len(channels)} channels")
        return {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }
    
    except Exception as e:
        logger.error(f"Error in /baseline/current endpoint: {e}")
        return {"error": str(e)}


@app.get("/health")
def health():
    try:
        return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Error in /health endpoint: {e}")
        return {"status": "error", "message": str(e)}
