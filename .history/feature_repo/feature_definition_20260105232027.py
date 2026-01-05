# feature_repo/feature_definition.py
from dataclasses import dataclass
from typing import List

LEAKAGE_COLS = {"fiber", "sugars", "calories"}

@dataclass(frozen=True)
class FeatureSpec:
    categorical: List[str]
    numerical: List[str]

    @property
    def all_features(self) -> List[str]:
        return self.numerical + self.categorical
