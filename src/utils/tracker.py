from pathlib import Path


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class ExperimentTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scalar_path = self.run_dir / "scalars.tsv"
        self.writer = SummaryWriter(log_dir=str(self.run_dir)) if SummaryWriter else None

        if not self.scalar_path.exists():
            self.scalar_path.write_text("step\ttag\tvalue\n", encoding="utf-8")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        with self.scalar_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{step}\t{tag}\t{value:.8f}\n")

        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
