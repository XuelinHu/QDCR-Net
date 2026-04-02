from dataclasses import dataclass


@dataclass
class Trainer:
    config: dict

    def fit(self) -> None:
        experiment = self.config.get("experiment", {}).get("name", "unknown")
        print(f"[train] placeholder trainer for experiment: {experiment}")

    def evaluate(self) -> None:
        experiment = self.config.get("experiment", {}).get("name", "unknown")
        print(f"[eval] placeholder evaluator for experiment: {experiment}")
