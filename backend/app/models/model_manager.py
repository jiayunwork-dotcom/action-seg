import json
from typing import Dict, Optional
from pathlib import Path

import torch

from app.core.config import settings
from app.models.feature_extractor import FeatureExtractor3D
from app.models.segmentation import MultiStageTCN


class ModelManager:
    _instance = None
    _models_cache: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._load_configs()

    def _load_configs(self):
        with open(settings.MODEL_CONFIG_PATH, "r") as f:
            self.model_config = json.load(f)

        with open(settings.ACTION_CLASSES_PATH, "r") as f:
            self.action_classes = json.load(f)["action_classes"]

        self.num_classes = len(self.action_classes)

    def get_action_classes(self) -> list:
        return self.action_classes

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_model_config(self, version: str = "latest") -> dict:
        if version not in self.model_config["models"]:
            version = "latest"
        return self.model_config["models"][version]

    def get_available_versions(self) -> list:
        return list(self.model_config["models"].keys())

    def get_models(self, version: str = "latest", device: str = "cpu") -> Dict:
        cache_key = f"{version}_{device}"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        config = self.get_model_config(version)
        feat_cfg = config["feature_extractor"]
        seg_cfg = config["segmentation"]

        feature_extractor = FeatureExtractor3D(
            feature_dim=feat_cfg["feature_dim"],
            random_seed=settings.RANDOM_SEED,
        )

        segmentation = MultiStageTCN(
            num_stages=seg_cfg["num_stages"],
            num_layers=seg_cfg["num_layers"],
            num_f_maps=seg_cfg["num_f_maps"],
            dim=feat_cfg["feature_dim"],
            num_classes=self.num_classes,
            dilations=seg_cfg.get("dilations", [1, 2, 4, 8]),
            random_seed=settings.RANDOM_SEED,
        )

        models = {
            "feature_extractor": feature_extractor,
            "segmentation": segmentation,
            "config": config,
        }

        self._models_cache[cache_key] = models
        return models
