import keras
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
import json

from src.types import RuntimeDataset
from core.config import config
from utils.logger import logger


class BaseAlgorithm(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model: keras.Model | None = None

    @abstractmethod
    def build(self, num_classes: int) -> keras.Model:
        """子類別實作：回傳編譯好的 Keras Model"""
        pass

    def train(self, data: RuntimeDataset):
        logger.info(
            f"[{self.name}] Initializing model for {data.num_classes} classes..."
        )
        self.model = self.build(data.num_classes)

        # 顯示架構
        self.model.summary(print_fn=lambda x: logger.debug(x))

        logger.info(f"[{self.name}] Starting Training ({config.EPOCHS} epochs)...")

        save_dir = config.DATA_DIR / "models" / self.name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint Callback
        ckpt = keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        )

        history = self.model.fit(
            data.train_ds,
            validation_data=data.val_ds,
            epochs=config.EPOCHS,
            callbacks=[ckpt],
        )

        self._save_plots(history, save_dir)
        self._save_results(history.history, save_dir)
        logger.info(f"[{self.name}] Training done. Saved to {save_dir}")

    def benchmark(self, data: RuntimeDataset):
        if self.model is None:
            logger.error(f"[{self.name}] Model is not trained yet.")
            return

        logger.info(f"[{self.name}] Starting Benchmarking...")

        results = self.model.evaluate(data.val_ds)
        logger.info(f"[{self.name}] Benchmarking done.")
        return {
            "loss": results[0],
            "accuracy": results[1],
        }

    def _save_plots(self, history, save_dir: Path):
        # 繪製 Accuracy
        plt.figure()
        plt.plot(history.history["accuracy"], label="Train")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Val")
        plt.title(f"{self.name} Accuracy")
        plt.ylabel("Acc")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(save_dir / "accuracy.png")
        plt.close()

        # 繪製 Loss
        plt.figure()
        plt.plot(history.history["loss"], label="Train")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Val")
        plt.title(f"{self.name} Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(save_dir / "loss.png")
        plt.close()

    def _save_results(self, results: dict, save_dir: Path):
        with open(save_dir / "results.json", "w") as f:
            best_index = results.get("val_accuracy", []).index(
                max(results.get("val_accuracy", [0]))
            )
            json.dump(
                {
                    "results": results,
                    "best_epoch": best_index + 1,
                    "best_val_accuracy": results.get("val_accuracy", [0])[best_index],
                    "best_val_loss": results.get("val_loss", [0])[best_index],
                },
                f,
                indent=4,
            )
