from typing import Dict, Type
from models.base import BaseAlgorithm

from models.algorithms.VGG16 import VGG16Algorithm
from models.algorithms.VGG19 import VGG19Algorithm
from models.algorithms.ResNet50 import ResNet50Algorithm
from models.algorithms.ResNet101 import ResNet101Algorithm
from models.algorithms.ResNet152 import ResNet152Algorithm
from models.algorithms.DenseNet121 import DenseNet121Algorithm
from models.algorithms.DenseNet169 import DenseNet169Algorithm
from models.algorithms.DenseNet201 import DenseNet201Algorithm
from models.algorithms.InceptionV3 import InceptionV3Algorithm
from models.algorithms.EfficientNetB6 import EfficientNetB6Algorithm

MODEL_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "VGG16": VGG16Algorithm,
    "VGG19": VGG19Algorithm,
    "ResNet50": ResNet50Algorithm,
    "ResNet101": ResNet101Algorithm,
    "ResNet152": ResNet152Algorithm,
    "DenseNet121": DenseNet121Algorithm,
    "DenseNet169": DenseNet169Algorithm,
    "DenseNet201": DenseNet201Algorithm,
    "InceptionV3": InceptionV3Algorithm,
    "EfficientNetB6": EfficientNetB6Algorithm,
}
