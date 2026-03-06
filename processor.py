#!/usr/bin/env python3
import json
import io
import logging
import shutil
import boto3
import pandas as pd
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector

s3 = boto3.client("s3")
logger = logging.getLogger(__name__)

NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]
LOG_FILE = "/opt/anomaly-detection/app.log"  # Same path as in app.py

def process_file(bucket: str, key: str):
    """Process a single CSV file: detect anomalies, update baseline, save results."""
    try:
        logger.info(f"========== Starting processing: s3://{bucket}/{key} ==========")

        # 1. Download raw file
        try:
            logger.info(f"Downloading file from S3: {key}")
            response = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to download/parse file {key}: {e}")
            raise

        # 2. Load current baseline
        try:
            baseline_mgr = BaselineManager(bucket=bucket)
            baseline = baseline_mgr.load()
            logger.info(f"Baseline loaded with {len([k for k in baseline.keys() if k != 'last_updated'])} channels")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            raise

        # 3. Update baseline with values from this batch BEFORE scoring
        try:
            logger.info("Updating baseline with new observations")
            for col in NUMERIC_COLS:
                if col in df.columns:
                    clean_values = df[col].dropna().tolist()
                    if clean_values:
                        baseline = baseline_mgr.update(baseline, col, clean_values)
                        logger.info(f"  {col}: added {len(clean_values)} values")
                    else:
                        logger.warning(f"  {col}: no valid values to add")
                else:
                    logger.warning(f"  {col}: column not found in data")
        except Exception as e:
            logger.error(f"Failed to update baseline: {e}")
            raise

        # 4. Run detection
        try:
            logger.info("Running anomaly detection")
            detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
            scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
            anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
            logger.info(f"Detection complete: {anomaly_count}/{len(df)} anomalies flagged")
        except Exception as e:
            logger.error(f"Failed to run anomaly detection: {e}")
            raise

        # 5. Write scored file to processed/ prefix
        try:
            output_key = key.replace("raw/", "processed/")
            logger.info(f"Writing scored file to s3://{bucket}/{output_key}")
            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType="text/csv"
            )
            logger.info(f"Scored file written successfully ({len(csv_buffer.getvalue())} bytes)")
        except Exception as e:
            logger.error(f"Failed to write scored file: {e}")
            raise

        # 6. Save updated baseline back to S3
        try:
            logger.info("Saving updated baseline to S3")
            baseline_mgr.save(baseline)
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
            raise

        # 7. Sync log file to S3 (IMPORTANT: required by project)
        try:
            log_s3_key = "logs/app.log"
            logger.info(f"Syncing log file to s3://{bucket}/{log_s3_key}")
            with open(LOG_FILE, 'rb') as log_file:
                s3.put_object(
                    Bucket=bucket,
                    Key=log_s3_key,
                    Body=log_file.read(),
                    ContentType="text/plain"
                )
            logger.info("Log file synced to S3 successfully")
        except FileNotFoundError:
            logger.warning(f"Log file not found at {LOG_FILE}, skipping sync")
        except Exception as e:
            logger.error(f"Failed to sync log file to S3: {e}")
            # Don't raise - log sync failure shouldn't stop processing

        # 8. Build and save processing summary
        try:
            summary = {
                "source_key": key,
                "output_key": output_key,
                "processed_at": datetime.utcnow().isoformat(),
                "total_rows": len(df),
                "anomaly_count": anomaly_count,
                "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
                "baseline_observation_counts": {
                    col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
                }
            }

            summary_key = output_key.replace(".csv", "_summary.json")
            logger.info(f"Writing summary to s3://{bucket}/{summary_key}")
            s3.put_object(
                Bucket=bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2),
                ContentType="application/json"
            )
            logger.info("Summary written successfully")
        except Exception as e:
            logger.error(f"Failed to write summary: {e}")
            raise

        logger.info(f"========== Processing complete: {anomaly_count}/{len(df)} anomalies ==========")
        return summary

    except Exception as e:
        logger.error(f"========== Processing FAILED for {key}: {e} ==========")
        raise