#!/usr/bin/env python3
import json
import math
import logging
import boto3
from datetime import datetime
from typing import Optional

s3 = boto3.client("s3")
logger = logging.getLogger(__name__)


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key
        logger.info(f"BaselineManager initialized for bucket: {bucket}, key: {baseline_key}")

    def load(self) -> dict:
        """Load baseline from S3, return empty dict if not found."""
        try:
            logger.info(f"Loading baseline from s3://{self.bucket}/{self.baseline_key}")
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info(f"Baseline loaded successfully with {len(baseline)} entries")
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.warning(f"Baseline file not found in S3, starting with empty baseline")
            return {}
        except Exception as e:
            logger.error(f"Error loading baseline from S3: {e}")
            return {}

    def save(self, baseline: dict):
        """Save baseline to S3."""
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()
            body = json.dumps(baseline, indent=2)
            
            logger.info(f"Saving baseline to s3://{self.bucket}/{self.baseline_key}")
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=body,
                ContentType="application/json"
            )
            logger.info(f"Baseline saved successfully ({len(body)} bytes)")
        except Exception as e:
            logger.error(f"Error saving baseline to S3: {e}")
            raise

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}
                logger.info(f"Initializing new channel in baseline: {channel}")

            state = baseline[channel]
            initial_count = state["count"]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state
            
            new_observations = state["count"] - initial_count
            logger.info(f"Updated baseline for {channel}: +{new_observations} observations "
                       f"(total: {state['count']}, mean: {state['mean']:.4f}, std: {state['std']:.4f})")
            
            return baseline
        
        except Exception as e:
            logger.error(f"Error updating baseline for channel {channel}: {e}")
            raise

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        """Get statistics for a specific channel."""
        return baseline.get(channel)