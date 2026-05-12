# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:43:32 2026

@author: LUCIANOGARIM
"""

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================
# 1) Building Blocks: Time Embedding and ResBlock
# ================================================


class SinusoidalTimeEmbedding(layers.Layer):
    """
    Encode the diffusion timestep into a dense sinusoidal vector.

    This layer follows the common sinusoidal positional/time embedding strategy
    used in diffusion models. The scalar timestep is projected into a vector
    containing sine and cosine components at multiple frequencies.
    """

    def __init__(self, dim):
        """
        Initialize the sinusoidal time embedding layer.

        Parameters
        ----------
        dim : int
            Output embedding dimension. The implementation uses half of the
            dimensions for sine terms and half for cosine terms.
        """
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb_scale = math.log(10000) / (self.half_dim - 1)

    def call(self, time):
        """
        Compute the sinusoidal embedding for a batch of timesteps.

        Parameters
        ----------
        time : tf.Tensor
            Timestep tensor with shape (batch_size, 1).

        Returns
        -------
        tf.Tensor
            Sinusoidal time embedding with shape (batch_size, dim).
        """
        # time shape: (batch_size, 1)
        emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb_scale)
        emb = time * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class ResidualBlock1D(layers.Layer):
    """
    1D convolutional residual block conditioned on the time embedding.

    The block normalizes and activates the input, applies convolutional layers,
    injects the projected time embedding, and uses a shortcut connection to
    preserve residual information.
    """

    def __init__(self, filters, kernel_size=3):
        """
        Initialize the residual block.

        Parameters
        ----------
        filters : int
            Number of output filters/channels.
        kernel_size : int, optional
            Kernel size used by the Conv1D layers.
        """
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding="same")
        self.conv2 = layers.Conv1D(filters, kernel_size, padding="same")
        self.norm1 = layers.GroupNormalization(groups=8)  # GN is better than BN for diffusion models.
        self.norm2 = layers.GroupNormalization(groups=8)
        self.act = layers.Activation("swish")  # Swish/SiLU is standard in diffusion architectures.
        self.time_proj = layers.Dense(filters, activation="swish")  # Project time embedding to filters.
        self.shortcut = layers.Conv1D(filters, 1, padding="same")

    def call(self, x, time_emb):
        """
        Apply the residual block.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor with shape (batch, steps, channels).
        time_emb : tf.Tensor
            Time embedding tensor with shape (batch, time_dim).

        Returns
        -------
        tf.Tensor
            Output tensor after residual processing.
        """
        # x: (batch, steps, channels)
        # time_emb: (batch, time_dim)

        h = self.conv1(self.act(self.norm1(x)))

        # Add the time embedding using broadcasting along the sequence axis.
        # time_emb projected: (batch, filters) -> (batch, 1, filters)
        t_vec = self.time_proj(time_emb)
        h = h + t_vec[:, tf.newaxis, :]

        h = self.conv2(self.act(self.norm2(h)))

        if x.shape[-1] != h.shape[-1]:
            x = self.shortcut(x)

        return x + h


# ================================================
# 2) 1D U-Net: Denoiser Model
# ================================================


def make_unet_1d(T, A, time_steps=1000):
    """
    Build a 1D U-Net denoiser for short well-log sequences.

    The model receives a noisy sequence and its diffusion timestep, then predicts
    the noise component that should be removed during denoising.

    Parameters
    ----------
    T : int
        Sequence length.
    A : int
        Number of attributes/logs per sequence step.
    time_steps : int, optional
        Number of diffusion timesteps. This argument is preserved for API
        compatibility, although the current architecture does not use it directly.

    Returns
    -------
    keras.Model
        1D U-Net diffusion denoiser.

    Input
    -----
    - input_x: shape (batch, T, A)
    - input_t: shape (batch, 1)

    Output
    ------
    - predicted noise with shape (batch, T, A)
    """

    # Inputs.
    input_x = layers.Input(shape=(T, A))
    input_t = layers.Input(shape=(1,))  # Integer timestep.

    # 1. Process time.
    time_dim = 64
    t_emb = SinusoidalTimeEmbedding(time_dim)(input_t)

    # MLP to process the raw time embedding.
    t_emb = layers.Dense(time_dim * 4, activation="swish")(t_emb)
    t_emb = layers.Dense(time_dim)(t_emb)

    # 2. Encoder / downsampling path.
    # Since T=20 is small, downsampling is done carefully: 20 -> 10 -> 5.

    # Initial convolution.
    x = layers.Conv1D(64, 3, padding="same")(input_x)

    # Block 1 at resolution 20.
    x1 = ResidualBlock1D(64)(x, t_emb)

    # Down 1: 20 -> 10.
    x = layers.Conv1D(128, 3, strides=2, padding="same")(x1)
    x2 = ResidualBlock1D(128)(x, t_emb)

    # Down 2: 10 -> 5.
    x = layers.Conv1D(256, 3, strides=2, padding="same")(x2)
    x3 = ResidualBlock1D(256)(x, t_emb)

    # 3. Bottleneck at resolution 5.
    mid = ResidualBlock1D(256)(x3, t_emb)
    mid = ResidualBlock1D(256)(mid, t_emb)

    # 4. Decoder / upsampling path with skip connections.

    # Up 1: 5 -> 10.
    x = layers.UpSampling1D(size=2)(mid)
    x = layers.Concatenate()([x, x2])
    x = ResidualBlock1D(128)(x, t_emb)

    # Up 2: 10 -> 20.
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Concatenate()([x, x1])
    x = ResidualBlock1D(64)(x, t_emb)

    # 5. Final output.
    # The model predicts noise with the same attribute dimension as the input.
    outputs = layers.Conv1D(A, 3, padding="same")(x)

    model = keras.Model(inputs=[input_x, input_t], outputs=outputs, name="unet_1d_diff")
    return model
