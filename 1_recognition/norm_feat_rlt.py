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
            
            if mean_key not in self.norm_stats or std_key not in self.norm_stats:
                raise KeyError(f"Mean or std not found for feature: {key}")
            
            mean = torch.tensor(self.norm_stats[mean_key], dtype=torch.float32)
            std = torch.tensor(self.norm_stats[std_key], dtype=torch.float32)
            
            normalized_value = (value - mean) / std
            normalized_features[key] = normalized_value
        
        return normalized_features