"""Parameterised model zoo: MLP, CNN, ResNet and CNN+BiLSTM families.

All models take (28, 28, 1) float images in [0, 1], end in a softmax over
``num_classes`` and are compiled with Adam + sparse categorical cross-entropy.
``hidden_layers`` always means the number of Dense(units) layers in the
classification head (for ``mlp`` that is the whole network body).
"""

from __future__ import annotations

import keras

INPUT_SHAPE = (28, 28, 1)
FAMILIES = ("mlp", "cnn", "resnet", "cnn_bilstm")

L = keras.layers


def _augmentation(x):
    """Light augmentation; Keras only applies these while training."""
    x = L.RandomRotation(0.05)(x)
    x = L.RandomTranslation(0.1, 0.1)(x)
    return L.RandomZoom(0.1)(x)


def _dense_head(x, hidden_layers: int, units: int, dropout: float):
    for _ in range(hidden_layers):
        x = L.Dense(units, activation="relu")(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.Dropout(dropout)(x)
    return x


def _conv_bn(x, filters: int, stride: int = 1):
    x = L.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    return L.BatchNormalization(momentum=0.9)(x)


def _residual_block(x, filters: int, stride: int):
    shortcut = x
    y = _conv_bn(x, filters, stride)
    y = L.Activation("relu")(y)
    y = _conv_bn(y, filters)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = L.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(x)
        shortcut = L.BatchNormalization(momentum=0.9)(shortcut)
    return L.Activation("relu")(L.Add()([y, shortcut]))


def _body_mlp(x, units, blocks, dropout):
    return L.Flatten()(x)


def _body_cnn(x, units, blocks, dropout):
    for filters in (32, 64):
        x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.MaxPooling2D()(x)
        x = L.Dropout(dropout * 0.5)(x)
    return L.Flatten()(x)


def _body_resnet(x, units, blocks, dropout):
    x = L.Activation("relu")(_conv_bn(x, 16))
    filters = 16
    for i in range(blocks):
        stride = 2 if i > 0 else 1
        if i > 0:
            filters *= 2
        x = _residual_block(x, filters, stride)
    return L.GlobalAveragePooling2D()(x)


def _body_cnn_bilstm(x, units, blocks, dropout):
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.MaxPooling2D()(x)  # 14 x 14 x 32
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.MaxPooling2D((2, 1))(x)  # 7 x 14 x 64 (height x width x channels)
    x = L.Permute((2, 1, 3))(x)  # width becomes the time axis
    x = L.Reshape((14, 7 * 64))(x)
    return L.Bidirectional(L.LSTM(max(units // 2, 1)))(x)


_BODIES = {
    "mlp": _body_mlp,
    "cnn": _body_cnn,
    "resnet": _body_resnet,
    "cnn_bilstm": _body_cnn_bilstm,
}


def build_model(
    family: str,
    num_classes: int,
    hidden_layers: int = 2,
    units: int = 128,
    blocks: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    augment: bool = True,
    seed: int | None = None,
) -> keras.Model:
    """Build and compile a model of the given family.

    ``blocks`` is only used by ``resnet``.  With ``augment=True`` every family
    gets augmentation layers, which are inactive at inference.  ``seed`` seeds
    Keras before the weights are built.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if hidden_layers < 0:
        raise ValueError(f"hidden_layers must be >= 0, got {hidden_layers}")
    if blocks < 1:
        raise ValueError(f"blocks must be >= 1, got {blocks}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    if seed is not None:
        keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = _augmentation(inputs) if augment else inputs
    x = _BODIES[family](x, units, blocks, dropout)
    x = _dense_head(x, hidden_layers, units, dropout)
    outputs = L.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"hcr_{family}")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model: keras.Model) -> int:
    """Total number of parameters (trainable and non-trainable)."""
    return int(model.count_params())
