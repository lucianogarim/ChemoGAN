"""Transformer-VAE baseline (revision 1).

Reviewer R1.6 noted that the first-submission baseline used beta = 1e-5 and sampled
the decoder directly from N(0, I), so that the prior and the aggregate posterior were
mismatched. Two remedies are now available and selected on validation wells:
  * KL warm-up to a larger beta (standard beta-VAE training);
  * ex-post density estimation (Ghosh et al., 2020): a Gaussian mixture is fitted to
    the encoder means of the training windows and new latents are drawn from it.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.mixture import GaussianMixture
import transformer_lib


class _BetaWarmup(keras.callbacks.Callback):
    def __init__(self, beta_max, warmup_epochs):
        super().__init__()
        self.beta_max = beta_max
        self.warmup = max(1, warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.beta.assign(self.beta_max * min(1.0, (epoch + 1) / self.warmup))


class TVAEGenerator:
    def __init__(self, vae, sampler, Z, gmm=None, seed=0):
        self.vae, self.sampler, self.Z, self.gmm = vae, sampler, Z, gmm
        self.rng = np.random.default_rng(seed)

    def generate(self, n):
        if self.sampler == "gmm":
            z, _ = self.gmm.sample(n)
            z = z[self.rng.permutation(n)].astype("float32")
        else:
            z = self.rng.standard_normal((n, self.Z)).astype("float32")
        return self.vae.decoder(z, training=False).numpy()


def train_transformer(X_train, T, A, Z_dim, epochs=100, batch_size=64, beta=1e-5,
                      warmup_epochs=0, sampler="prior", gmm_components=10, seed=0, patience=15):
    embed_dim = 64
    encoder = transformer_lib.make_transformer_encoder(T, A, embed_dim, Z_dim)
    decoder = transformer_lib.make_transformer_generator(T, A, Z_dim, embed_dim)
    vae = transformer_lib.TransformerVAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    cbs = [keras.callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)]
    if warmup_epochs > 0:
        cbs.append(_BetaWarmup(beta, warmup_epochs))
    print(f"===== Transformer-VAE (beta={beta}, warmup={warmup_epochs}, sampler={sampler}) =====")
    hist = vae.fit(X_train, epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=0)
    print(f"   stopped after {len(hist.history['loss'])} epochs, loss={hist.history['loss'][-1]:.4f}")
    gmm = None
    if sampler == "gmm":
        mu, _, _ = encoder.predict(X_train, batch_size=1024, verbose=0)
        gmm = GaussianMixture(gmm_components, covariance_type="full", random_state=seed).fit(mu)
    return TVAEGenerator(vae, sampler, Z_dim, gmm, seed), hist.history
