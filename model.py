# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionV3


def build_embedding_model(input_shape=(224, 224, 3), embedding_dim=256, base_trainable=False, dropout_rate=0.0):
    """
    Builds a CNN model with InceptionV3 base and a custom projection head for embedding.
    
    Args:
        input_shape: Shape of input images.
        embedding_dim: Dimension of output embedding.
        base_trainable: If True, fine-tunes the InceptionV3 base.

    Returns:
        A compiled Keras model that outputs L2-normalized embeddings.
    """
    # Load InceptionV3 without the classification head
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = base_trainable

    # Add global pooling and dense projection layer
    x = layers.Input(shape=input_shape)
    y = base_model(x, training=base_trainable)
    y = layers.GlobalAveragePooling2D()(y)
    if dropout_rate and dropout_rate > 0:
        y = layers.Dropout(dropout_rate, name="proj_dropout")(y)
    y = layers.Dense(embedding_dim)(y)  # No activation

    model = Model(inputs=x, outputs=y, name="embedding_model")
    model.base_model = base_model
    return model
