import keras
from keras import layers, models, applications, optimizers
from models.base import BaseAlgorithm
from core.config import config


class EfficientNetB6Algorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__("EfficientNetB6")

    def build(self, num_classes: int) -> keras.Model:
        base = applications.EfficientNetB6(
            include_top=False, weights="imagenet", input_shape=config.IMG_SIZE + (3,)
        )
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = models.Model(inputs=base.input, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
