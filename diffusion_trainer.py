import tensorflow as tf
from tensorflow import keras
import numpy as np
import diffusion_lib
from tqdm import tqdm


class DiffusionManager:
    """
    Manager class for training and sampling from a 1D diffusion model.

    This class stores the diffusion schedule, performs one training step by
    adding noise to real samples, and generates synthetic samples through the
    reverse denoising process.
    """

    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initialize the diffusion manager and create the linear noise schedule.

        Parameters
        ----------
        model : keras.Model
            Neural network used to predict the noise added at each timestep.
        timesteps : int, optional
            Number of diffusion timesteps.
        beta_start : float, optional
            Initial beta value for the linear schedule.
        beta_end : float, optional
            Final beta value for the linear schedule.
        """
        self.model = model
        self.clip_lo, self.clip_hi = -1.0, 1.0
        self._predict = tf.function(lambda x, t: self.model([x, t], training=False))
        self.timesteps = timesteps

        # Diffusion parameters using a linear schedule.
        self.beta = np.linspace(beta_start, beta_end, timesteps).astype(np.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

        # Convert constants to float32 tensors to avoid dtype errors during training.
        self.sqrt_alpha_bar = tf.constant(np.sqrt(self.alpha_bar), dtype=tf.float32)
        self.sqrt_one_minus_alpha_bar = tf.constant(np.sqrt(1.0 - self.alpha_bar), dtype=tf.float32)

    @tf.function
    def train_step(self, x_batch, optimizer):
        """
        Run one diffusion-model training step.

        The method samples random timesteps, adds the corresponding amount of
        Gaussian noise to each input sample, and trains the model to predict the
        noise that was added.

        Parameters
        ----------
        x_batch : tf.Tensor
            Batch of real samples with shape (batch, T, A).
        optimizer : keras.optimizers.Optimizer
            Optimizer used to update the diffusion model weights.

        Returns
        -------
        tf.Tensor
            Scalar mean squared error loss for the current batch.
        """
        # Explicitly cast the input to float32.
        x_batch = tf.cast(x_batch, dtype=tf.float32)
        batch_size = tf.shape(x_batch)[0]

        # 1. Sample timesteps.
        t = tf.random.uniform((batch_size, 1), minval=0, maxval=self.timesteps, dtype=tf.int32)

        # 2. Generate Gaussian noise using float32.
        noise = tf.random.normal(tf.shape(x_batch), dtype=tf.float32)

        # 3. Retrieve alpha parameters for timestep t.
        sqrt_ab_t = tf.gather(self.sqrt_alpha_bar, t)[:, 0]
        sqrt_ab_t = tf.reshape(sqrt_ab_t, [batch_size, 1, 1])

        sqrt_one_minus_ab_t = tf.gather(self.sqrt_one_minus_alpha_bar, t)[:, 0]
        sqrt_one_minus_ab_t = tf.reshape(sqrt_one_minus_ab_t, [batch_size, 1, 1])

        # 4. Create the noisy sample.
        x_noisy = sqrt_ab_t * x_batch + sqrt_one_minus_ab_t * noise

        with tf.GradientTape() as tape:
            # 5. Predict the noise.
            noise_pred = self.model([x_noisy, tf.cast(t, tf.float32)], training=True)

            # 6. Manual MSE loss to avoid Keras AttributeError issues.
            # Difference -> square -> global mean.
            loss = tf.reduce_mean(tf.square(noise - noise_pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def generate(self, num_samples, T, A):
        """
        Generate synthetic samples through the reverse denoising process.

        Parameters
        ----------
        num_samples : int
            Number of synthetic samples to generate.
        T : int
            Sequence length.
        A : int
            Number of attributes/logs per sequence step.

        Returns
        -------
        numpy.ndarray
            Generated samples with shape (num_samples, T, A).
        """
        print(f"Generating {num_samples} samples via reverse diffusion...")

        # Start from pure float32 Gaussian noise.
        x = tf.random.normal((num_samples, T, A), dtype=tf.float32)

        # Reverse denoising loop.
        for i in reversed(range(0, self.timesteps)):
            t_vec = tf.ones((num_samples, 1), dtype=tf.float32) * i

            # Predict the noise at the current timestep.
            pred_noise = self._predict(x, t_vec)

            # Posterior mean computed from the clipped x0 estimate (Ho et al., 2020, reference
            # implementation, clip_denoised=True): keeps the reverse process inside the data range.
            ab = self.alpha_bar[i]
            ab_prev = self.alpha_bar[i - 1] if i > 0 else 1.0
            a_t, b_t = self.alpha[i], self.beta[i]
            x0 = (x - np.sqrt(1.0 - ab) * pred_noise) / np.sqrt(ab)
            x0 = tf.clip_by_value(x0, self.clip_lo, self.clip_hi)
            c0 = b_t * np.sqrt(ab_prev) / (1.0 - ab)
            ct = (1.0 - ab_prev) * np.sqrt(a_t) / (1.0 - ab)
            mean = c0 * x0 + ct * x
            if i > 0:
                var = b_t * (1.0 - ab_prev) / (1.0 - ab)  # posterior variance beta_tilde
                x = mean + np.sqrt(var) * tf.random.normal(tf.shape(x), dtype=tf.float32)
            else:
                x = mean
        x = x.numpy()
        if getattr(self, "rescale", False):
            x = (x + 1.0) / 2.0
        return x


def train_diffusion(X_train, T, A, epochs=100, batch_size=64, rescale=True, log_every=50):
    """
    Train a 1D diffusion model and return its manager.

    Parameters
    ----------
    X_train : array-like
        Training data with shape (n_samples, T, A).
    T : int
        Sequence length.
    A : int
        Number of attributes/logs per sequence step.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Training batch size.

    Returns
    -------
    DiffusionManager
        Trained diffusion manager containing the model and diffusion schedule.
    """
    # Instantiate the model.
    unet = diffusion_lib.make_unet_1d(T, A)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    # Instantiate the diffusion manager.
    manager = DiffusionManager(unet)

    # Build the training dataset.
    # Revision 1: DDPMs assume data on a scale comparable to the N(0, I) prior; the
    # MinMax [0, 1] windows are mapped to [-1, 1] for training and mapped back after sampling.
    manager.rescale = rescale
    Xt = (2.0 * X_train - 1.0) if rescale else X_train
    train_ds = tf.data.Dataset.from_tensor_slices(Xt.astype("float32")).shuffle(4096).batch(batch_size)

    # Exponential moving average of the weights (decay 0.999), used for sampling as in Ho et al. (2020).
    ema = diffusion_lib.make_unet_1d(T, A)
    ema.set_weights(unet.get_weights())
    ema_decay = 0.999

    @tf.function
    def ema_update():
        for w_e, w in zip(ema.weights, unet.weights):
            w_e.assign(ema_decay * w_e + (1.0 - ema_decay) * w)

    print(f"\n===== Starting Diffusion Model Training ({epochs} epochs) =====")

    for epoch in range(epochs):
        loss_mean = keras.metrics.Mean()

        for batch in train_ds:
            loss = manager.train_step(batch, optimizer)
            ema_update()
            loss_mean.update_state(loss)

        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {loss_mean.result():.5f}")

    manager.model = ema
    manager._predict = tf.function(lambda x, t: ema([x, t], training=False))
    manager.clip_lo, manager.clip_hi = (-1.0, 1.0) if rescale else (0.0, 1.0)
    return manager
