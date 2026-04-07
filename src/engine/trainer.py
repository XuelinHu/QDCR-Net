from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import UnderwaterDetectionDataset
from src.engine.detection_ops import compute_detection_metrics, decode_predictions, greedy_match, pairwise_iou_xywh
from src.losses import DetectionLoss
from src.models import BaselineDetector, QDCRNet
from src.utils import ExperimentTracker, get_logger


@dataclass
class Trainer:
    config: dict

    def __post_init__(self) -> None:
        self.logger = get_logger("qdcr.trainer")
        experiment_config = self.config.get("experiment", {})
        self.output_dir = Path(experiment_config.get("output_dir", "outputs/checkpoints/qdcr_net_smoke"))
        self.run_dir = Path(experiment_config.get("runs_dir", f"runs/{experiment_config.get('name', 'default')}"))
        self.checkpoint_path = self.output_dir / "latest.pt"
        self.best_checkpoint_path = self.output_dir / "best.pt"
        self.prediction_path = self.output_dir / "predictions.json"
        self.metrics_path = self.output_dir / "metrics.json"
        self.device = torch.device(
            experiment_config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self._set_seed(int(experiment_config.get("seed", 42)))

    def fit(self) -> None:
        train_dataset, val_dataset = self._build_datasets()
        model = self._build_model().to(self.device)
        loss_fn = DetectionLoss().to(self.device)
        train_config = self.config.get("train", {})
        eval_config = self.config.get("eval", {})
        early_stopping = train_config.get("early_stopping", {})
        optimizer = self._build_optimizer(model)
        tracker = ExperimentTracker(self.run_dir)
        batch_size = min(int(train_config.get("batch_size", 4)), len(train_dataset))
        eval_batch_size = min(int(train_config.get("batch_size", 4)), len(val_dataset))
        max_batches = int(train_config.get("max_batches_per_epoch", 2))
        eval_max_batches = int(eval_config.get("max_batches", 2))
        epochs = int(train_config.get("epochs", 1))
        patience = int(early_stopping.get("patience", 5))
        min_delta = float(early_stopping.get("min_delta", 1e-4))
        monitor = str(early_stopping.get("monitor", "loss"))
        mode = str(early_stopping.get("mode", "min")).lower()
        enabled = bool(early_stopping.get("enabled", True))
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(batch_size, 1),
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(eval_batch_size, 1),
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_batch,
        )
        best_metric = float("inf") if mode == "min" else float("-inf")
        epochs_without_improvement = 0
        best_epoch = 0

        self.logger.info(
            "[train] experiment=%s dataset_size=%s batch_size=%s epochs=%s max_batches=%s device=%s early_stopping=%s monitor=%s mode=%s patience=%s min_delta=%.6f",
            self.config.get("experiment", {}).get("name", "unknown"),
            len(train_dataset),
            batch_size,
            epochs,
            max_batches,
            self.device,
            enabled,
            monitor,
            mode,
            patience,
            min_delta,
        )

        try:
            if self.checkpoint_path.exists():
                try:
                    metadata = model.load_checkpoint(
                        self.checkpoint_path,
                        optimizer=optimizer,
                        map_location=self.device,
                    )
                    self.logger.info(
                        "[train] resumed checkpoint=%s metadata=%s",
                        self.checkpoint_path,
                        metadata,
                    )
                except RuntimeError as exc:
                    self.logger.warning(
                        "[train] skipped incompatible checkpoint=%s error=%s",
                        self.checkpoint_path,
                        exc,
                    )

            global_step = 0
            for epoch_index in range(epochs):
                model.train()
                epoch_loss = 0.0
                epoch_box_iou = 0.0
                epoch_correct = 0
                sample_count = 0

                for batch_index, batch in enumerate(train_loader):
                    if max_batches > 0 and batch_index >= max_batches:
                        break

                    raw_image = batch["raw_image"].to(self.device)
                    enhanced_image = batch["enhanced_image"].to(self.device)
                    predictions = model(raw_image, enhanced_image)
                    matched_classes, matched_boxes, matched_mask = self._match_batch(predictions, batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss_output = loss_fn(predictions, matched_classes, matched_boxes, matched_mask)
                    loss = loss_output["loss"]
                    loss.backward()
                    optimizer.step()

                    batch_iou = self._mean_iou(loss_output["pred_boxes"], matched_boxes, matched_mask)
                    batch_correct = self._count_class_matches(
                        loss_output["predicted_class"],
                        matched_classes,
                        matched_mask,
                    )
                    batch_size_value = int(matched_mask.sum().item())
                    batch_loss_value = float(loss.item())
                    batch_acc_value = batch_correct / max(batch_size_value, 1)

                    epoch_loss += batch_loss_value * batch_size_value
                    epoch_box_iou += batch_iou * batch_size_value
                    epoch_correct += batch_correct
                    sample_count += batch_size_value
                    global_step += 1

                    tracker.log_scalar("train/batch_loss", batch_loss_value, global_step)
                    tracker.log_scalar("train/batch_acc", batch_acc_value, global_step)
                    tracker.log_scalar("train/batch_box_iou", batch_iou, global_step)
                    tracker.log_scalar(
                        "train/batch_cls_loss",
                        float(loss_output["classification_loss"].item()),
                        global_step,
                    )
                    tracker.log_scalar(
                        "train/batch_box_loss",
                        float(loss_output["box_loss"].item()),
                        global_step,
                    )
                    self.logger.info(
                        "[train] epoch=%s batch=%s loss=%.4f acc=%.2f iou=%.2f objects=%s",
                        epoch_index + 1,
                        batch_index + 1,
                        batch_loss_value,
                        batch_acc_value,
                        batch_iou,
                        batch_size_value,
                    )

                epoch_loss_value = epoch_loss / max(sample_count, 1)
                epoch_acc_value = epoch_correct / max(sample_count, 1)
                epoch_iou_value = epoch_box_iou / max(sample_count, 1)
                tracker.log_scalar("train/epoch_loss", epoch_loss_value, epoch_index + 1)
                tracker.log_scalar("train/epoch_acc", epoch_acc_value, epoch_index + 1)
                tracker.log_scalar("train/epoch_box_iou", epoch_iou_value, epoch_index + 1)
                model.save_checkpoint(
                    self.checkpoint_path,
                    optimizer=optimizer,
                    metadata={
                        "experiment": self.config.get("experiment", {}).get("name", "unknown"),
                        "epoch": epoch_index + 1,
                        "samples": sample_count,
                        "device": str(self.device),
                    },
                )
                val_metrics = self._evaluate_model(
                    model=model,
                    loss_fn=loss_fn,
                    val_loader=val_loader,
                    max_batches=eval_max_batches,
                    conf_thresh=float(eval_config.get("conf_thresh", 0.25)),
                    iou_thresh=float(eval_config.get("iou_thresh", 0.5)),
                    tracker=tracker,
                    tracker_step=epoch_index + 1,
                )
                monitored_value = float(val_metrics[monitor])
                tracker.log_scalar(f"train/early_stop_{monitor}", monitored_value, epoch_index + 1)
                improved = self._is_improved(
                    current=monitored_value,
                    best=best_metric,
                    mode=mode,
                    min_delta=min_delta,
                )
                if improved:
                    best_metric = monitored_value
                    best_epoch = epoch_index + 1
                    epochs_without_improvement = 0
                    model.save_checkpoint(
                        self.best_checkpoint_path,
                        optimizer=optimizer,
                        metadata={
                            "experiment": self.config.get("experiment", {}).get("name", "unknown"),
                            "epoch": epoch_index + 1,
                            "samples": sample_count,
                            "device": str(self.device),
                            "best_metric": best_metric,
                            "monitor": monitor,
                        },
                    )
                else:
                    epochs_without_improvement += 1
                self.logger.info(
                    "[train] epoch=%s summary loss=%.4f acc=%.2f iou=%.2f val_%s=%.4f best=%.4f best_epoch=%s wait=%s/%s samples=%s",
                    epoch_index + 1,
                    epoch_loss_value,
                    epoch_acc_value,
                    epoch_iou_value,
                    monitor,
                    monitored_value,
                    best_metric,
                    best_epoch,
                    epochs_without_improvement,
                    patience,
                    sample_count,
                )
                if enabled and epochs_without_improvement >= patience:
                    self.logger.info(
                        "[train] early stopping triggered at epoch=%s best_epoch=%s best_%s=%.4f",
                        epoch_index + 1,
                        best_epoch,
                        monitor,
                        best_metric,
                    )
                    break
        finally:
            tracker.close()

    def evaluate(self) -> dict[str, float]:
        _, val_dataset = self._build_datasets()
        model = self._build_model().to(self.device)
        loss_fn = DetectionLoss().to(self.device)
        tracker = ExperimentTracker(self.run_dir)
        batch_size = min(int(self.config.get("train", {}).get("batch_size", 4)), len(val_dataset))
        max_batches = int(self.config.get("eval", {}).get("max_batches", 2))
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(batch_size, 1),
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

        try:
            eval_checkpoint = self._resolve_eval_checkpoint()
            if eval_checkpoint is not None:
                try:
                    metadata = model.load_checkpoint(eval_checkpoint, map_location=self.device)
                    self.logger.info("[eval] loaded checkpoint=%s metadata=%s", eval_checkpoint, metadata)
                except RuntimeError as exc:
                    self.logger.warning(
                        "[eval] skipped incompatible checkpoint=%s error=%s",
                        eval_checkpoint,
                        exc,
                    )
            else:
                self.logger.warning("[eval] checkpoint not found, using fresh model=%s", self.checkpoint_path)

            self.logger.info(
                "[eval] experiment=%s dataset_size=%s batch_size=%s max_batches=%s device=%s",
                self.config.get("experiment", {}).get("name", "unknown"),
                len(val_dataset),
                batch_size,
                max_batches,
                self.device,
            )

            model.eval()
            eval_config = self.config.get("eval", {})
            metrics = self._evaluate_model(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                max_batches=max_batches,
                conf_thresh=float(eval_config.get("conf_thresh", 0.25)),
                iou_thresh=float(eval_config.get("iou_thresh", 0.5)),
                tracker=tracker,
                tracker_step=1,
            )
            complexity_metrics = self._complexity_metrics(model, self._last_profile_batch)
            speed_metrics = self._speed_metrics(
                model=model,
                dataset=val_dataset,
                conf_thresh=float(eval_config.get("conf_thresh", 0.25)),
                iou_thresh=float(eval_config.get("iou_thresh", 0.5)),
                max_batches=min(max_batches, 10),
                batch_size=min(int(self.config.get("train", {}).get("batch_size", 4)), len(val_dataset)),
            )
            metrics.update(complexity_metrics)
            metrics.update(speed_metrics)
            tracker.log_scalar("eval/params_m", metrics["params_m"], 1)
            tracker.log_scalar("eval/gflops", metrics["gflops"], 1)
            tracker.log_scalar("eval/fps", metrics["fps"], 1)
            self._write_eval_artifacts(self._last_predictions, metrics)
            self.logger.info(
                "[eval] summary loss=%.4f acc=%.2f iou=%.2f map50=%.3f map50_95=%.3f params=%.3fM gflops=%.3f fps=%.2f samples=%s",
                metrics["loss"],
                metrics["acc"],
                metrics["box_iou"],
                metrics["map50"],
                metrics["map50_95"],
                metrics["params_m"],
                metrics["gflops"],
                metrics["fps"],
                self._last_eval_samples,
            )
            return metrics
        finally:
            tracker.close()

    @property
    def _background_class(self) -> int:
        return int(self.config.get("dataset", {}).get("num_classes", 4))

    def _build_datasets(self) -> tuple[UnderwaterDetectionDataset, UnderwaterDetectionDataset]:
        dataset_config = self.config.get("dataset", {})
        common_args = {
            "num_classes": int(dataset_config.get("num_classes", 4)),
            "image_size": int(dataset_config.get("image_size", 128)),
            "synthetic_size": int(dataset_config.get("synthetic_size", 16)),
            "max_objects": int(dataset_config.get("max_objects", 8)),
        }
        train_root = self._as_path(dataset_config.get("train_root"))
        val_root = self._as_path(dataset_config.get("val_root"))
        train_dataset = UnderwaterDetectionDataset(
            image_root=train_root,
            annotation_root=self._derive_annotation_root(train_root),
            enhanced_root=self._as_path(dataset_config.get("enhanced_root")),
            **common_args,
        )
        val_dataset = UnderwaterDetectionDataset(
            image_root=val_root,
            annotation_root=self._derive_annotation_root(val_root),
            enhanced_root=self._as_path(dataset_config.get("enhanced_root")),
            **common_args,
        )
        return train_dataset, val_dataset

    def _build_model(self) -> torch.nn.Module:
        dataset_config = self.config.get("dataset", {})
        model_config = self.config.get("model", {})
        model_name = str(model_config.get("name", "qdcr_net")).lower()
        model_cls = BaselineDetector if "baseline" in model_name else QDCRNet
        return model_cls(
            num_classes=int(dataset_config.get("num_classes", 4)),
            feature_dim=int(model_config.get("feature_dim", 32)),
            num_queries=int(model_config.get("num_queries", dataset_config.get("max_objects", 8))),
        )

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        train_config = self.config.get("train", {})
        optimizer_name = str(train_config.get("optimizer", "adamw")).lower()
        learning_rate = float(train_config.get("lr", 1e-3))
        weight_decay = float(train_config.get("weight_decay", 0.0))
        if optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _derive_annotation_root(self, image_root: Path | None) -> Path | None:
        if image_root is None:
            return None
        if image_root.name == "images":
            return image_root.parent / "labels"
        return image_root / "labels"

    def _as_path(self, value: str | None) -> Path | None:
        return None if value is None else Path(value)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _collate_batch(self, batch: list[dict]) -> dict[str, object]:
        return {
            "sample_id": [sample["sample_id"] for sample in batch],
            "raw_image": torch.stack([sample["raw_image"] for sample in batch], dim=0),
            "enhanced_image": torch.stack([sample["enhanced_image"] for sample in batch], dim=0),
            "gt_boxes": [sample["gt_boxes"] for sample in batch],
            "gt_classes": [sample["gt_classes"] for sample in batch],
            "source": [sample["source"] for sample in batch],
        }

    def _match_batch(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = predictions["logits"]
        pred_boxes = predictions["pred_boxes"]
        gt_classes_list: list[torch.Tensor] = batch["gt_classes"]  # type: ignore[assignment]
        gt_boxes_list: list[torch.Tensor] = batch["gt_boxes"]  # type: ignore[assignment]

        matched_classes_list = []
        matched_boxes_list = []
        matched_mask_list = []
        for batch_index in range(logits.size(0)):
            gt_classes = gt_classes_list[batch_index].to(self.device)
            gt_boxes = gt_boxes_list[batch_index].to(self.device)
            matched_classes, matched_boxes, matched_mask = greedy_match(
                logits[batch_index],
                pred_boxes[batch_index],
                gt_classes,
                gt_boxes,
                background_class=self._background_class,
            )
            matched_classes_list.append(matched_classes)
            matched_boxes_list.append(matched_boxes)
            matched_mask_list.append(matched_mask)

        return (
            torch.stack(matched_classes_list, dim=0),
            torch.stack(matched_boxes_list, dim=0),
            torch.stack(matched_mask_list, dim=0),
        )

    def _mean_iou(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> float:
        iou = pairwise_iou_xywh(
            pred_boxes[valid_mask],
            target_boxes[valid_mask],
        )
        if iou.numel() == 0:
            return 0.0
        diagonal = torch.diag(iou)
        return float(diagonal.mean().item()) if diagonal.numel() else 0.0

    def _count_class_matches(
        self,
        predicted_class: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> int:
        valid_predictions = predicted_class[valid_mask]
        valid_targets = targets[valid_mask]
        if valid_targets.numel() == 0:
            return 0
        return int((valid_predictions == valid_targets).sum().item())

    def _ground_truth_records(self, batch: dict[str, object]) -> list[dict[str, torch.Tensor]]:
        gt_classes_list: list[torch.Tensor] = batch["gt_classes"]  # type: ignore[assignment]
        gt_boxes_list: list[torch.Tensor] = batch["gt_boxes"]  # type: ignore[assignment]
        return [
            {
                "classes": gt_classes.cpu(),
                "boxes": gt_boxes.cpu(),
            }
            for gt_classes, gt_boxes in zip(gt_classes_list, gt_boxes_list)
        ]

    def _write_eval_artifacts(
        self,
        predictions: list[dict[str, torch.Tensor]],
        metrics: dict[str, float],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        serializable_predictions = []
        for image_index, prediction in enumerate(predictions):
            serializable_predictions.append(
                {
                    "image_index": image_index,
                    "scores": prediction["scores"].cpu().tolist(),
                    "classes": prediction["classes"].cpu().tolist(),
                    "boxes": prediction["boxes"].cpu().tolist(),
                }
            )

        self.prediction_path.write_text(
            json.dumps(serializable_predictions, indent=2),
            encoding="utf-8",
        )
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _resolve_eval_checkpoint(self) -> Path | None:
        eval_config = self.config.get("eval", {})
        checkpoint_mode = str(eval_config.get("checkpoint", "best")).lower()
        if checkpoint_mode == "latest":
            return self.checkpoint_path if self.checkpoint_path.exists() else None
        if self.best_checkpoint_path.exists():
            return self.best_checkpoint_path
        if self.checkpoint_path.exists():
            return self.checkpoint_path
        return None

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        loss_fn: DetectionLoss,
        val_loader: DataLoader,
        max_batches: int,
        conf_thresh: float,
        iou_thresh: float,
        tracker: ExperimentTracker | None = None,
        tracker_step: int | None = None,
    ) -> dict[str, float]:
        total_loss = 0.0
        total_iou = 0.0
        total_correct = 0
        sample_count = 0
        all_predictions: list[dict[str, torch.Tensor]] = []
        all_ground_truths: list[dict[str, torch.Tensor]] = []
        profile_batch: dict[str, torch.Tensor] | None = None

        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break

                raw_image = batch["raw_image"].to(self.device)
                enhanced_image = batch["enhanced_image"].to(self.device)
                if profile_batch is None:
                    profile_batch = {
                        "raw_image": raw_image[:1],
                        "enhanced_image": enhanced_image[:1],
                    }
                predictions = model(raw_image, enhanced_image)
                matched_classes, matched_boxes, matched_mask = self._match_batch(predictions, batch)
                loss_output = loss_fn(predictions, matched_classes, matched_boxes, matched_mask)

                batch_size_value = int(matched_mask.sum().item())
                total_loss += float(loss_output["loss"].item()) * batch_size_value
                total_iou += self._mean_iou(loss_output["pred_boxes"], matched_boxes, matched_mask) * batch_size_value
                total_correct += self._count_class_matches(
                    loss_output["predicted_class"],
                    matched_classes,
                    matched_mask,
                )
                sample_count += batch_size_value

                decoded = decode_predictions(
                    predictions["logits"],
                    predictions["pred_boxes"],
                    background_class=self._background_class,
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh,
                )
                all_predictions.extend(decoded)
                all_ground_truths.extend(self._ground_truth_records(batch))

        metrics = {
            "loss": total_loss / max(sample_count, 1),
            "acc": total_correct / max(sample_count, 1),
            "box_iou": total_iou / max(sample_count, 1),
        }
        metrics.update(
            compute_detection_metrics(
                all_predictions,
                all_ground_truths,
                num_classes=int(self.config.get("dataset", {}).get("num_classes", 4)),
            )
        )
        self._last_predictions = all_predictions
        self._last_ground_truths = all_ground_truths
        self._last_profile_batch = profile_batch
        self._last_eval_samples = sample_count
        if tracker is not None and tracker_step is not None:
            tracker.log_scalar("eval/loss", metrics["loss"], tracker_step)
            tracker.log_scalar("eval/acc", metrics["acc"], tracker_step)
            tracker.log_scalar("eval/box_iou", metrics["box_iou"], tracker_step)
            tracker.log_scalar("eval/map50", metrics["map50"], tracker_step)
            tracker.log_scalar("eval/map50_95", metrics["map50_95"], tracker_step)
        return metrics

    def _is_improved(self, current: float, best: float, mode: str, min_delta: float) -> bool:
        if mode == "max":
            return current > best + min_delta
        return current < best - min_delta

    def _complexity_metrics(
        self,
        model: torch.nn.Module,
        profile_batch: dict[str, torch.Tensor] | None,
    ) -> dict[str, float]:
        params = float(sum(parameter.numel() for parameter in model.parameters()))
        gflops = 0.0
        if profile_batch is not None:
            gflops = self._estimate_gflops(
                model,
                profile_batch["raw_image"],
                profile_batch["enhanced_image"],
            )
        return {
            "params_m": params / 1_000_000.0,
            "gflops": gflops,
        }

    def _estimate_gflops(
        self,
        model: torch.nn.Module,
        raw_image: torch.Tensor,
        enhanced_image: torch.Tensor,
    ) -> float:
        total_flops = 0.0
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal total_flops
            input_tensor = inputs[0]
            batch_size = float(input_tensor.shape[0])
            out_height = float(output.shape[2])
            out_width = float(output.shape[3])
            kernel_ops = float(module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups))
            output_elements = batch_size * float(module.out_channels) * out_height * out_width
            total_flops += output_elements * kernel_ops * 2.0

        def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal total_flops
            input_tensor = inputs[0]
            batch_elements = float(input_tensor.numel() / max(module.in_features, 1))
            total_flops += batch_elements * float(module.in_features) * float(module.out_features) * 2.0

        for submodule in model.modules():
            if isinstance(submodule, nn.Conv2d):
                handles.append(submodule.register_forward_hook(conv_hook))
            elif isinstance(submodule, nn.Linear):
                handles.append(submodule.register_forward_hook(linear_hook))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(raw_image, enhanced_image)
        if was_training:
            model.train()

        for handle in handles:
            handle.remove()

        return total_flops / 1_000_000_000.0

    def _speed_metrics(
        self,
        model: torch.nn.Module,
        dataset: UnderwaterDetectionDataset,
        conf_thresh: float,
        iou_thresh: float,
        max_batches: int,
        batch_size: int,
    ) -> dict[str, float]:
        if len(dataset) == 0 or max_batches <= 0:
            return {"fps": 0.0}

        loader = DataLoader(
            dataset,
            batch_size=max(batch_size, 1),
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

        was_training = model.training
        model.eval()
        timed_images = 0
        timed_batches = 0
        start_time = 0.0

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                raw_image = batch["raw_image"].to(self.device)
                enhanced_image = batch["enhanced_image"].to(self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                batch_start = time.perf_counter()
                predictions = model(raw_image, enhanced_image)
                decode_predictions(
                    predictions["logits"],
                    predictions["pred_boxes"],
                    background_class=self._background_class,
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                batch_elapsed = time.perf_counter() - batch_start
                if batch_index == 0:
                    continue
                start_time += batch_elapsed
                timed_images += int(raw_image.size(0))
                timed_batches += 1

        if was_training:
            model.train()

        if timed_batches == 0 or start_time <= 0.0:
            return {"fps": 0.0}
        return {"fps": timed_images / start_time}
