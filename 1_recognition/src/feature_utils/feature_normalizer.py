import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import csv
import torch
import datetime
from tqdm import tqdm


class NormRealTime:
    def __init__(self, norm_stats_path: str,feature_keys=None):
        self.norm_stats_path = norm_stats_path
        self.norm_stats = self._load_norm_stats()
        # np.load on an .npz returns a LAZY NpzFile: every __getitem__ re-reads and
        # re-inflates that array from the zip archive. normalize_features() runs once
        # per frame and touches a mean+std per feature group, so leaving the stats in
        # the NpzFile re-inflated all 32 arrays every frame -- measured 4.1 ms/frame,
        # ~6% of the whole live loop, for values that never change after load.
        # Materialize them once as tensors instead (0.002 ms/frame).
        self._stats = {
            key: torch.as_tensor(np.asarray(self.norm_stats[key]), dtype=torch.float32)
            for key in self.norm_stats.files
        }

    def _load_norm_stats(self) -> dict:
        if not os.path.exists(self.norm_stats_path):
            raise FileNotFoundError(f"Norm stats file not found: {self.norm_stats_path}")

        norm_stats = np.load(self.norm_stats_path,allow_pickle=True)

        return norm_stats

    def normalize_features(self, features: dict) -> dict:
        normalized_features = {}
        for key, value in features.items():
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            
            if mean_key not in self._stats or std_key not in self._stats:
                raise KeyError(f"Mean or std not found for feature: {key}")

            mean = self._stats[mean_key]
            std = self._stats[std_key]
            # value may be a plain numpy array (e.g. StreamingH36MFeatureExtractor's
            # output) or already a torch.Tensor -- numpy.ndarray - torch.Tensor isn't
            # reliably supported, so normalize to a tensor first either way.
            value = torch.as_tensor(value, dtype=torch.float32)

            normalized_value = (value - mean) / std
            normalized_features[key] = normalized_value
        
        return normalized_features