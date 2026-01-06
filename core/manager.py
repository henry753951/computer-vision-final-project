from rich.table import Table
from core.registry import MODEL_REGISTRY
from src.dataloader import ImageLoader
from utils.logger import logger, console


class CLIManager:
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
