import pickle
import tensorflow as tf
from typing import List, Tuple, Dict

from core.config import config
from src.types import CachedMetadata, RuntimeDataset
from utils.logger import logger


class ImageLoader:
    # ==========================
    # Part 1: 純文字解析 (Private)
    # ==========================
    @staticmethod
    def _parse_raw_files() -> Tuple[List[str], List[str], List[int]]:
        """
        解析 name.txt 與 query.txt
        回傳: (all_paths, all_labels_str, query_indices)
        """
        if not config.NAME_PATH.exists():
            raise FileNotFoundError(f"Name file not found: {config.NAME_PATH}")

        # 1. 解析 name.txt
        full_paths: List[str] = []
        labels: List[str] = []

        with open(config.NAME_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                # 格式: Filename Label
                filename = parts[0]
                label_name = " ".join(parts[1:])

                # 組合實體路徑: data/pictures/{Label}/{Filename}
                # 注意：這裡轉為 str 以便 pickle
                p = config.PICTURES_DIR / label_name / filename
                full_paths.append(str(p))
                labels.append(label_name)

        # 2. 解析 query.txt (Indices)
        q_indices: List[int] = []
        if config.QUERY_PATH.exists():
            with open(config.QUERY_PATH, "r") as f:
                for line in f:
                    if line.strip().isdigit():
                        q_indices.append(int(line.strip()))

        return full_paths, labels, q_indices

    @classmethod
    def _generate_metadata(cls) -> CachedMetadata:
        """
        執行解析、編碼與切分，並產生可快取的 Metadata 物件
        """
        logger.info("[yellow]Parsing raw text files (No cache found)...[/]")
        all_paths, all_labels_str, query_indices_list = cls._parse_raw_files()

        total_count = len(all_paths)
        if total_count == 0:
            raise ValueError("No images found in name.txt")

        # Label Encoding
        unique_classes = sorted(list(set(all_labels_str)))
        class_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(unique_classes)
        }
        all_labels_int = [class_to_idx[l] for l in all_labels_str]

        # Splitting
        query_set = set(query_indices_list)

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []

        for i in range(total_count):
            if i in query_set:
                val_paths.append(all_paths[i])
                val_labels.append(all_labels_int[i])
            else:
                train_paths.append(all_paths[i])
                train_labels.append(all_labels_int[i])

        return CachedMetadata(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            class_names=unique_classes,
            num_classes=len(unique_classes),
        )

    # ==========================
    # Part 2: TF Dataset 建構
    # ==========================
    @staticmethod
    def _decode_image(
        path: tf.Tensor, label: tf.Tensor, num_classes: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """TF Graph 內部的圖片讀取函數"""
        img_raw = tf.io.read_file(path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, config.IMG_SIZE)
        img = tf.cast(img, tf.float32)
        # img = img / 255.0  (視模型需求決定是否在此做 Rescaling)

        label_one_hot = tf.one_hot(label, depth=num_classes)
        return img, label_one_hot

    @classmethod
    def load_dataset(cls, force_refresh: bool = False) -> RuntimeDataset:
        """
        主入口：檢查快取 -> 載入 Metadata -> 轉換為 TF Dataset
        """

        # 1. 取得 Metadata (從快取或重新解析)
        metadata: CachedMetadata

        if not force_refresh and config.CACHE_PATH.exists():
            try:
                with open(config.CACHE_PATH, "rb") as f:
                    metadata = pickle.load(f)
                logger.info(
                    f"[green]Loaded metadata from cache[/]: {config.CACHE_PATH}"
                )
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), regenerating...")
                metadata = cls._generate_metadata()
        else:
            metadata = cls._generate_metadata()
            # 寫入快取
            try:
                with open(config.CACHE_PATH, "wb") as f:
                    pickle.dump(metadata, f)
                logger.info(f"Cache saved to {config.CACHE_PATH}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        # 2. 建構 TensorFlow Datasets
        logger.info(
            f"Building TF Pipelines. Train: {len(metadata.train_paths)}, Val: {len(metadata.val_paths)}"
        )

        def build_pipe(paths: List[str], labels: List[int]) -> tf.data.Dataset:
            if not paths:
                # 回傳空 Dataset 防止報錯
                return tf.data.Dataset.from_tensors(
                    (tf.zeros(config.IMG_SIZE + (3,)), tf.zeros(metadata.num_classes))
                ).take(0)

            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.shuffle(len(paths), seed=7414) if len(paths) > 0 else ds
            ds = ds.map(
                lambda p, l: cls._decode_image(p, l, metadata.num_classes),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            ds = ds.batch(config.BATCH_SIZE)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = build_pipe(metadata.train_paths, metadata.train_labels)
        val_ds = build_pipe(metadata.val_paths, metadata.val_labels)

        return RuntimeDataset(
            train_ds=train_ds,
            val_ds=val_ds,
            train_steps=len(metadata.train_paths) // config.BATCH_SIZE,
            val_steps=len(metadata.val_paths) // config.BATCH_SIZE,
            num_classes=metadata.num_classes,
            class_names=metadata.class_names,
        )
