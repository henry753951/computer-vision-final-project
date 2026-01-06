from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    BASE_DIR: Path = Path(".")
    DATA_DIR: Path = BASE_DIR / "data"

    # 資料路徑
    # 圖片路徑格式: data/pictures/{Label}/{Filename}
    PICTURES_DIR: Path = DATA_DIR / "pictures"

    NAME_PATH: Path = DATA_DIR / "name.txt"
    QUERY_PATH: Path = DATA_DIR / "query.txt"

    # 快取檔位置
    CACHE_PATH: Path = DATA_DIR / "dataset_metadata.pkl"

    # 訓練參數
    IMG_SIZE: tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4


config = AppConfig()
