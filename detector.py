#!/usr/bin/env python3
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional

logger = logging.getLogger(__name__)


class AnomalyDetector:

    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        self.z_threshold = z_threshold
        self.contamination = contamination
        logger.info(f"AnomalyDetector initialized (z_threshold={z_threshold}, contamination={contamination})")

    def zscore_flag(
        self,
        values: pd.Series,
        mean: float,
        std: float
    ) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        try:
            if std == 0:
                logger.warning("Standard deviation is 0, returning zeros for z-scores")
                return pd.Series([0.0] * len(values))
            z_scores = (values - mean).abs() / std
            logger.debug(f"Computed z-scores: min={z_scores.min():.4f}, max={z_scores.max():.4f}, mean={z_scores.mean():.4f}")
            return z_scores
        except Exception as e:
            logger.error(f"Error computing z-scores: {e}")
            raise

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        Scores closer to -1 indicate stronger anomalies.
        """
        try:
            logger.info(f"Running IsolationForest on {len(numeric_cols)} columns with {len(df)} rows")
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Fill missing values with median
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            logger.debug(f"Data prepared: shape={X.shape}, missing values filled")
            
            model.fit(X)
            logger.debug("IsolationForest model fitted")

            labels = model.predict(X)          # -1 = anomaly, 1 = normal
            scores = model.decision_function(X)  # lower = more anomalous
            
            anomaly_count = (labels == -1).sum()
            logger.info(f"IsolationForest detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")

            return labels, scores
        
        except Exception as e:
            logger.error(f"Error in IsolationForest detection: {e}")
            raise

    def run(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        baseline: dict,
        method: str = "both"
    ) -> pd.DataFrame:
        """
        Run anomaly detection on the dataframe using z-score, IsolationForest, or both.
        """
        try:
            logger.info(f"Running anomaly detection with method='{method}' on {len(df)} rows")
            result = df.copy()

            # --- Z-score per channel ---
            if method in ("zscore", "both"):
                logger.info("Running z-score detection per channel")
                for col in numeric_cols:
                    try:
                        stats = baseline.get(col)
                        if stats and stats["count"] >= 30:  # need enough history to trust baseline
                            logger.debug(f"  {col}: using baseline (mean={stats['mean']:.4f}, std={stats['std']:.4f})")
                            z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                            result[f"{col}_zscore"] = z_scores.round(4)
                            result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                            flagged_count = (z_scores > self.z_threshold).sum()
                            logger.info(f"  {col}: {flagged_count} values exceed z-score threshold")
                        else:
                            # Not enough baseline history yet — flag as unknown
                            count = stats["count"] if stats else 0
                            logger.warning(f"  {col}: insufficient baseline history (count={count}, need 30)")
                            result[f"{col}_zscore"] = None
                            result[f"{col}_zscore_flag"] = None
                    except Exception as e:
                        logger.error(f"  Error processing z-score for {col}: {e}")
                        result[f"{col}_zscore"] = None
                        result[f"{col}_zscore_flag"] = None

            # --- IsolationForest across all channels ---
            if method in ("isolation", "both"):
                try:
                    logger.info("Running IsolationForest detection across all channels")
                    labels, scores = self.isolation_forest_flag(df, numeric_cols)
                    result["if_label"] = labels          # -1 or 1
                    result["if_score"] = scores.round(4) # continuous anomaly score
                    result["if_flag"] = labels == -1
                except Exception as e:
                    logger.error(f"IsolationForest failed, setting flags to False: {e}")
                    result["if_label"] = 1
                    result["if_score"] = 0.0
                    result["if_flag"] = False

            # --- Consensus flag: anomalous by at least one method ---
            if method == "both":
                try:
                    logger.info("Computing consensus anomaly flag")
                    zscore_flags = [
                        result[f"{col}_zscore_flag"]
                        for col in numeric_cols
                        if f"{col}_zscore_flag" in result.columns
                        and result[f"{col}_zscore_flag"].notna().any()
                    ]
                    if zscore_flags:
                        any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                        result["anomaly"] = any_zscore | result["if_flag"]
                        logger.info(f"Consensus: combining z-score and IsolationForest flags")
                    else:
                        result["anomaly"] = result["if_flag"]
                        logger.info(f"Consensus: using only IsolationForest flags (no z-score baselines ready)")
                    
                    total_anomalies = result["anomaly"].sum()
                    logger.info(f"Final result: {total_anomalies} total anomalies flagged ({total_anomalies/len(df)*100:.2f}%)")
                except Exception as e:
                    logger.error(f"Error computing consensus flag: {e}")
                    result["anomaly"] = False

            return result
        
        except Exception as e:
            logger.error(f"Error in anomaly detection run: {e}")
            raise