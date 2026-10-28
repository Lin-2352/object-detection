"""Model architectures: a small CNN and the original tutorial MLP baseline."""

from __future__ import annotations

import keras

INPUT_SHAPE = (28, 28, 1)


def build_mlp(num_classes: int) -> keras.Model:
    """The original tutorial architecture, kept as a measurable baseline."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="hcr_mlp")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_symbol_cnn(num_classes: int) -> keras.Model:
    """Small CNN for stable, printed 28x28 glyph templates."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="hcr_symbol_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(num_classes: int, augment: bool = True) -> keras.Model:
    """A small, CPU-friendly CNN (~470k params) with light data augmentation.

    The augmentation layers (RandomRotation/Translation/Zoom) are only
    active while training: Keras runs them with training=False during
    predict/evaluate, so they have no effect at inference time.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = inputs
    if augment:
        x = keras.layers.RandomRotation(0.05)(x)
        x = keras.layers.RandomTranslation(0.1, 0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)

    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="hcr_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
