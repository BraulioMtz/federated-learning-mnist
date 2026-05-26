"""
TheModel.py — Modelo global para Aprendizaje Federado con MNIST
================================================================
Arquitectura: Mini-ResNet con bloques residuales
  - Distinta a la vista en clase (MLP / CNN simple)
  - Usa bloques residuales con BatchNormalization para mayor
    estabilidad durante el entrenamiento federado distribuido.

Uso:
    from TheModel import build_model, get_model_config

    model = build_model()
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
# Bloque residual
# ─────────────────────────────────────────────
def residual_block(x, filters: int, kernel_size: int = 3, stride: int = 1):
    """
    Bloque residual estilo ResNet:
        Conv -> BN -> ReLU -> Conv -> BN -> (+shortcut) -> ReLU

    Si stride > 1 o los canales cambian, se proyecta el shortcut.
    """
    shortcut = x

    # Primera convolución
    x = layers.Conv2D(
        filters, kernel_size,
        strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Segunda convolución
    x = layers.Conv2D(
        filters, kernel_size,
        strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # Proyección del shortcut si las dimensiones difieren
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, kernel_size=1,
            strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# ─────────────────────────────────────────────
# Construcción del modelo
# ─────────────────────────────────────────────
def build_model(
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 10,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Construye y compila un Mini-ResNet para MNIST.

    Arquitectura:
        Input (28×28×1)
        → Conv2D(32, 3×3) + BN + ReLU
        → ResBlock(32)
        → ResBlock(64, stride=2)   # 14×14
        → ResBlock(64)
        → ResBlock(128, stride=2)  # 7×7
        → GlobalAveragePooling
        → Dropout(0.4)
        → Dense(128, ReLU) + BN
        → Dense(10, softmax)

    Returns
    -------
    keras.Model
        Modelo compilado listo para entrenamiento.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Stem: extracción inicial de características
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Bloques residuales
    x = residual_block(x, filters=32)            # 28×28×32
    x = residual_block(x, filters=64, stride=2)  # 14×14×64
    x = residual_block(x, filters=64)            # 14×14×64
    x = residual_block(x, filters=128, stride=2) # 7×7×128

    # Cabeza de clasificación
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="MiniResNet_MNIST")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# Configuración serializable (útil para el servidor federado)
# ─────────────────────────────────────────────
def get_model_config() -> dict:
    """
    Devuelve los hiperparámetros del modelo como diccionario,
    para que el servidor federado pueda registrarlos.
    """
    return {
        "architecture": "MiniResNet",
        "input_shape": (28, 28, 1),
        "num_classes": 10,
        "blocks": [
            {"filters": 32, "stride": 1},
            {"filters": 64, "stride": 2},
            {"filters": 64, "stride": 1},
            {"filters": 128, "stride": 2},
        ],
        "head_units": 128,
        "dropout": 0.4,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
    }


# ─────────────────────────────────────────────
# Prueba rápida al ejecutar directamente
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model()
    model.summary()
    print("\nConfiguración del modelo:")
    import json
    print(json.dumps(get_model_config(), indent=2))
