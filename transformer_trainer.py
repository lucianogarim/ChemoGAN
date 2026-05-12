# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:22:24 2026

@author: LUCIANOGARIM
"""

import tensorflow as tf
from tensorflow import keras
import transformer_lib


def train_transformer(X_train, T, A, Z_dim, epochs=100, batch_size=64):
    """
    Configure and train the Transformer VAE.

    This function builds the Transformer encoder and decoder, wraps them inside
    the TransformerVAE class, trains the VAE on the provided training data, and
    returns the decoder model. The decoder is returned because it acts as the
    generator during synthetic sequence generation.

    Parameters
    ----------
    X_train : array-like or tf.Tensor
        Training sequences with shape (n_samples, T, A).
    T : int
        Sequence length.
    A : int
        Number of attributes/logs per sequence step.
    Z_dim : int
        Latent-space dimension.
    epochs : int, optional
        Maximum number of training epochs.
    batch_size : int, optional
        Training batch size.

    Returns
    -------
    keras.Model
        Trained Transformer decoder, used as the generator.
    """

    # Transformer hyperparameters.
    embed_dim = 64  # Attention vector dimension.

    # Instantiate networks.
    encoder = transformer_lib.make_transformer_encoder(T, A, embed_dim, Z_dim)
    decoder = transformer_lib.make_transformer_generator(T, A, Z_dim, embed_dim)

    # Instantiate the VAE.
    vae = transformer_lib.TransformerVAE(encoder, decoder)

    # Compile the model.
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    # Callbacks.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=15,
        restore_best_weights=True,
    )

    print(f"\n===== Starting Transformer VAE Training ({epochs} epochs) =====")
    vae.fit(
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
    )

    # Return the decoder because it is used as the generator.
    return decoder
