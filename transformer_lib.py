import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================
# 1) Transformer Building Blocks
# ================================================


class TransformerBlock(layers.Layer):
    """
    Transformer encoder block with multi-head self-attention and a feed-forward network.

    The block applies self-attention, dropout, residual connections, layer
    normalization, and a position-wise feed-forward network.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initialize the Transformer block.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension used by the attention layer and feed-forward output.
        num_heads : int
            Number of attention heads.
        ff_dim : int
            Hidden dimension of the feed-forward network.
        rate : float, optional
            Dropout rate.
        """
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """
        Apply the Transformer block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor with shape (batch, T, embed_dim).
        training : bool, optional
            Whether the layer is being called in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as the input.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class Sampling(layers.Layer):
    """
    Sample a latent vector using the reparameterization trick.

    The layer receives ``z_mean`` and ``z_log_var`` and returns a sampled latent
    vector ``z``. This allows gradients to flow through the stochastic sampling
    operation during VAE training.
    """

    def call(self, inputs):
        """
        Sample from the latent Gaussian distribution.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Tuple containing ``z_mean`` and ``z_log_var``.

        Returns
        -------
        tf.Tensor
            Sampled latent vector.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ================================================
# 2) Encoder and Decoder Definitions
# ================================================


def make_transformer_encoder(T, A, embed_dim, latent_dim):
    """
    Build the Transformer-based VAE encoder.

    Parameters
    ----------
    T : int
        Sequence length.
    A : int
        Number of attributes/logs per timestep.
    embed_dim : int
        Transformer embedding dimension.
    latent_dim : int
        Latent-space dimension.

    Returns
    -------
    keras.Model
        Encoder model that returns ``z_mean``, ``z_log_var``, and sampled ``z``.
    """
    inputs = layers.Input(shape=(T, A))

    # Initial dense projection to the embedding dimension.
    x = layers.Dense(embed_dim)(inputs)

    # Positional embedding.
    positions = tf.range(start=0, limit=T, delta=1)
    pos_emb = layers.Embedding(input_dim=T, output_dim=embed_dim)(positions)
    x = x + pos_emb

    # Transformer blocks.
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)

    # Global average pooling to condense the temporal dimension.
    x = layers.GlobalAveragePooling1D()(x)

    # Latent layers.
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def make_transformer_generator(T, A, latent_dim, embed_dim):
    """
    Build the Transformer decoder used as the generator.

    Parameters
    ----------
    T : int
        Sequence length.
    A : int
        Number of attributes/logs generated per timestep.
    latent_dim : int
        Latent-space dimension.
    embed_dim : int
        Transformer embedding dimension.

    Returns
    -------
    keras.Model
        Decoder/generator model that maps latent vectors to sequences with
        shape (T, A).
    """
    inputs = layers.Input(shape=(latent_dim,))

    # Map latent vector Z to the sequence embedding space (T * embed_dim).
    x = layers.Dense(T * embed_dim)(inputs)
    x = layers.Reshape((T, embed_dim))(x)

    # Add positional encoding again.
    positions = tf.range(start=0, limit=T, delta=1)
    pos_emb = layers.Embedding(input_dim=T, output_dim=embed_dim)(positions)
    x = x + pos_emb

    # Transformer blocks.
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)

    # Final sigmoid output for MinMax-normalized data in the 0-1 range.
    outputs = layers.Dense(A, activation="sigmoid")(x)

    decoder = keras.Model(inputs, outputs, name="generator_transformer")
    return decoder


# ================================================
# 3) Main VAE Class
# ================================================


class TransformerVAE(keras.Model):
    """
    Transformer-based variational autoencoder for sequential well-log data.

    The model combines a Transformer encoder, a Transformer decoder/generator,
    and a VAE objective composed of reconstruction loss plus a weighted KL term.
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        Initialize the Transformer VAE.

        Parameters
        ----------
        encoder : keras.Model
            Encoder model returning ``z_mean``, ``z_log_var``, and sampled ``z``.
        decoder : keras.Model
            Decoder/generator model that reconstructs sequences from ``z``.
        **kwargs
            Additional keyword arguments passed to ``keras.Model``.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """
        Return tracked metrics so Keras can reset them between epochs.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Run one VAE training step.

        The reconstruction loss is computed manually as MSE to avoid Keras
        attribute errors in this workflow. The KL term is weighted by a small
        beta factor to prioritize reconstruction quality.

        Parameters
        ----------
        data : tf.Tensor
            Input batch with shape (batch, T, A).

        Returns
        -------
        dict
            Current values of total loss, reconstruction loss, and KL loss.
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Manual MSE calculation to avoid Keras attribute errors.
            # 1. Squared difference.
            squared_diff = tf.square(data - reconstruction)

            # 2. Mean over features (axis=-1), returning shape (batch, T).
            mse_per_timestep = tf.reduce_mean(squared_diff, axis=-1)

            # 3. Sum across time (axis=1), returning shape (batch,).
            sum_mse_time = tf.reduce_sum(mse_per_timestep, axis=1)

            # 4. Batch mean, returning a scalar.
            reconstruction_loss = tf.reduce_mean(sum_mse_time)

            # KL loss: Kullback-Leibler divergence.
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # KL beta weight. A small value prioritizes reconstruction quality.
            total_loss = reconstruction_loss + 0.00001 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
