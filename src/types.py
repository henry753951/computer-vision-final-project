from numpy import test
import tensorflow as tf
from dataclasses import dataclass
from typing import List


@dataclass
class CachedMetadata:
    """
    這是一個純 Python 物件，可以被 Pickle 序列化存成檔案。
    存放已經解析、對齊、切分好的路徑與標籤索引。
    """

    train_paths: List[str]
    train_labels: List[int]

    val_paths: List[str]
    val_labels: List[int]

    test_paths: List[str]
    test_labels: List[int]

    class_names: List[str]
    num_classes: int


@dataclass
class RuntimeDataset:
    """
    這是 TensorFlow 執行時需要的物件，包含建立好的 Pipeline。
    """

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset

    train_steps: int
    val_steps: int
    test_steps: int

    num_classes: int
    class_names: List[str]
