import os
from rich.table import Table
from core.registry import MODEL_REGISTRY
from src.dataloader import ImageLoader
from utils.logger import logger, console


class CLIManager:
    def get_supported_models(self):
        return list(MODEL_REGISTRY.keys())

    def list_models(self):
        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")
        for name in MODEL_REGISTRY.keys():
            table.add_row(name)
        console.print(table)

    def train_model(self, model_name: str, force_refresh: bool):
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Model '{model_name}' not found.")
            return

        try:
            # 1. 載入資料 (含 Pickle 快取機制)
            dataset = ImageLoader.load_dataset(force_refresh=force_refresh)

            # 2. 實例化模型
            ModelClass = MODEL_REGISTRY[model_name]
            model_instance = ModelClass()

            # 3. 執行訓練
            model_instance.train(dataset)

        except Exception as e:
            logger.exception(f"Training Failed: {e}")

    def benchmark_model(self, model_name: str):
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Model '{model_name}' not found.")
            return
        # best.keras is not exist
        if os.path.exists(f"data/models/{model_name}/best.keras") is False:
            logger.error(f"Model weights for '{model_name}' not found.")
            return

        try:
            # 1. 載入資料 (含 Pickle 快取機制)
            dataset = ImageLoader.load_dataset(force_refresh=False)

            # 2. 實例化模型
            ModelClass = MODEL_REGISTRY[model_name]
            model_instance = ModelClass()

            model_instance.model = model_instance.build(dataset.num_classes)
            model_instance.model.load_weights(f"data/models/{model_name}/best.keras")

            # 3. 執行基準測試
            result = model_instance.benchmark(dataset)
            logger.info(f"Benchmark Result for {model_name}: {result}")

            return result

        except Exception as e:
            logger.exception(f"Benchmarking Failed: {e}")

    def benchmark_all_models(self):
        results = {}
        for model_name in MODEL_REGISTRY.keys():
            logger.info(f"Starting benchmark for model: {model_name}")
            result = self.benchmark_model(model_name)
            if result is None:
                logger.info("\n")
                continue
            results[model_name] = result
            logger.info(f"Completed benchmark for model: {model_name}")
            logger.info("\n")

        self.logging_results(results)

    def logging_results(self, results: dict):
        table = Table(title="Benchmark Results")
        table.add_column("Model Name", style="cyan")
        table.add_column("Validation Accuracy", style="green")
        table.add_column("Validation Loss", style="red")

        for model_name, result in results.items():
            table.add_row(
                model_name,
                f"{result['val_accuracy']:.4f}",
                f"{result['val_loss']:.4f}",
            )

        console.print(table)
