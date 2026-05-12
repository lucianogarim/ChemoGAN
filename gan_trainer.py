import tensorflow as tf
from tensorflow import keras
import numpy as np
import gan_lib


class ChemoGANTrainer:
    """
    Trainer class for the physics-informed ChemoGAN model.

    This class coordinates adversarial training between the generator and the
    discriminator while adding physics-informed penalties to the generator loss.
    The penalty terms encourage smoother generated curves, NMR hierarchy
    consistency, and expected correlations between selected well-log attributes.
    """

    def __init__(self, generator, discriminator, features, Z_dim):
        """
        Initialize the ChemoGAN trainer.

        Parameters
        ----------
        generator : keras.Model
            Generator model that maps latent vectors to synthetic well-log windows.
        discriminator : keras.Model
            Discriminator model that separates real windows from synthetic windows.
        features : list of str
            Ordered list of feature names. The order must match the last dimension
            of the training tensors.
        Z_dim : int
            Latent-space dimension used to sample random noise for the generator.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.features = features
        self.Z = Z_dim

        # Optimizers
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
        self.d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)

        # Penalty weights and hyperparameters
        self.attr_weights = np.ones(len(features), dtype=np.float32)
        self.SUAVIDADE = 1
        self.TV_ORDER = 2
        self.TV_HUBER_DELTA = None
        self.NMR_RESTRICAO = 2
        self.NPHI_RHOB = 2
        self.NPHI_DT = 2
        # self.PE_CA_WEIGHT = 2  # Weight for calibrating PE vs DWCA correlation.
        self.PE_SI_WEIGHT = 2

        # Feature index mapping used by the physics-informed penalties.
        try:
            self.idx_nmre = features.index('NMRE_FINAL')
            self.idx_nmrfl = features.index('NMRFL_FINAL')
            self.idx_nmrt = features.index('NMRT_FINAL')
            self.i_nphi = features.index('NPHI')
            self.i_rhob = features.index('RHOB')
            # self.idx_ca = features.index('DWCA')
            self.i_dt = features.index('DT')
            self.idx_pe = features.index('PE')
            self.idx_si = features.index('DWSI')
        except ValueError as e:
            print(f"Warning: A feature required for a penalty term was not found: {e}")

    def train_step(self, real_batch):
        """
        Run one training step for both generator and discriminator.

        Parameters
        ----------
        real_batch : tf.Tensor
            Batch of real windows with shape (batch, T, A).

        Returns
        -------
        tuple
            Generator loss, discriminator loss, adversarial generator loss,
            total-variation penalty, NMR penalty, RHOB-NPHI correlation penalty,
            DT-NPHI correlation penalty, and PE-SI correlation penalty.
        """
        batch_size = tf.shape(real_batch)[0]
        noise = tf.random.normal([batch_size, self.Z])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_batch = self.generator(noise, training=True)

            # Discriminator outputs for real and synthetic samples.
            real_out = self.discriminator(real_batch, training=True)
            fake_out = self.discriminator(fake_batch, training=True)

            d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
            d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
            d_loss = d_loss_real + d_loss_fake

            # Physics-informed penalties from gan_lib.
            tv_term = gan_lib.tv_penalty_btA(
                fake_batch,
                self.attr_weights,
                self.TV_ORDER,
                self.TV_HUBER_DELTA,
            )
            pen_nphi_rhob = gan_lib.corr_penalty(
                fake_batch[..., self.i_nphi],
                fake_batch[..., self.i_rhob],
                sign=-1.0,
            )
            pen_nphi_dt = gan_lib.corr_penalty(
                fake_batch[..., self.i_nphi],
                fake_batch[..., self.i_dt],
                sign=+1.0,
            )
            nmr_pen = gan_lib.nmr_chain_penalty(
                fake_batch,
                self.idx_nmrt,
                self.idx_nmre,
                self.idx_nmrfl,
            )

            # Penalizes the PE-Si relationship if the expected negative correlation is not met.
            # loss_pe_ca = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_ca], sign=+1.0)
            loss_pe_si = gan_lib.corr_penalty(
                fake_batch[..., self.idx_pe],
                fake_batch[..., self.idx_si],
                sign=-1.0,
            )

            # Generator adversarial loss.
            g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)

            g_loss = (
                g_loss_adv
                + self.SUAVIDADE * tv_term
                + self.NMR_RESTRICAO * nmr_pen
                + self.NPHI_RHOB * pen_nphi_rhob
                + self.NPHI_DT * pen_nphi_dt
                # + self.PE_CA_WEIGHT * loss_pe_ca
                + self.PE_SI_WEIGHT * loss_pe_si
            )

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return g_loss, d_loss, g_loss_adv, tv_term, nmr_pen, pen_nphi_rhob, pen_nphi_dt, loss_pe_si

    @tf.function
    def val_step(self, val_batch):
        """
        Run one validation step without updating model weights.

        Parameters
        ----------
        val_batch : tf.Tensor
            Batch of validation windows with shape (batch, T, A).

        Returns
        -------
        tuple
            Validation generator loss and discriminator loss.
        """
        batch_size = tf.shape(val_batch)[0]
        noise = tf.random.normal([batch_size, self.Z])
        fake_batch = self.generator(noise, training=False)

        real_out = self.discriminator(val_batch, training=False)
        fake_out = self.discriminator(fake_batch, training=False)

        d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
        d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        d_loss = d_loss_real + d_loss_fake

        # Physics-informed penalties.
        tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
        pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
        pen_nphi_dt = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt], sign=+1.0)
        nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)

        # loss_pe_ca = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_ca], sign=+1.0)
        loss_pe_si = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_si], sign=-1.0)

        g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)

        g_loss = (
            g_loss_adv
            + self.SUAVIDADE * tv_term
            + self.NMR_RESTRICAO * nmr_pen
            + self.NPHI_RHOB * pen_nphi_rhob
            + self.NPHI_DT * pen_nphi_dt
            # + self.PE_CA_WEIGHT * loss_pe_ca
            + self.PE_SI_WEIGHT * loss_pe_si
        )

        return g_loss, d_loss

    def fit(self, train_ds, val_ds, epochs):
        """
        Train the ChemoGAN model for a fixed number of epochs.

        Parameters
        ----------
        train_ds : tf.data.Dataset
            Training dataset yielding real batches.
        val_ds : tf.data.Dataset
            Validation dataset yielding real batches.
        epochs : int
            Number of epochs to train.

        Returns
        -------
        dict
            Training history containing generator, discriminator, penalty, and
            validation losses for each epoch.
        """
        print(f"Starting training for {epochs} epochs...")

        # Training history. The key name `pe_ca_loss` is preserved for compatibility,
        # although the current stored value corresponds to the PE-SI penalty.
        history = {
            "g_loss": [],
            "d_loss": [],
            "g_adv": [],
            "tv_loss": [],
            "nmr_loss": [],
            "corr_rhob_nphi": [],
            "corr_dt_nphi": [],
            "pe_ca_loss": [],
            "val_g_loss": [],
            "val_d_loss": [],
        }

        for epoch in range(1, epochs + 1):
            ep_g, ep_d, ep_adv = [], [], []
            ep_tv, ep_nmr, ep_c1, ep_c2, ep_pe_ca = [], [], [], [], []

            for real_batch in train_ds:
                gl, dl, adv, tv, nmr, c1, c2, pe_ca = self.train_step(real_batch)

                ep_g.append(gl)
                ep_d.append(dl)
                ep_adv.append(adv)
                ep_tv.append(tv)
                ep_nmr.append(nmr)
                ep_c1.append(c1)
                ep_c2.append(c2)
                ep_pe_ca.append(pe_ca)

            val_g, val_d = [], []
            for val_batch in val_ds:
                vgl, vdl = self.val_step(val_batch)
                val_g.append(vgl)
                val_d.append(vdl)

            # Save epoch statistics.
            history["g_loss"].append(np.mean(ep_g))
            history["d_loss"].append(np.mean(ep_d))
            history["g_adv"].append(np.mean(ep_adv))
            history["tv_loss"].append(np.mean(ep_tv))
            history["nmr_loss"].append(np.mean(ep_nmr))
            history["corr_rhob_nphi"].append(np.mean(ep_c1))
            history["corr_dt_nphi"].append(np.mean(ep_c2))
            history["pe_ca_loss"].append(np.mean(ep_pe_ca))
            history["val_g_loss"].append(np.mean(val_g))
            history["val_d_loss"].append(np.mean(val_d))

            # Clean training log focused on the most relevant losses.
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"G_Loss: {history['g_loss'][-1]:.4f} | D_Loss: {history['d_loss'][-1]:.4f} | "
                f"Val G_Loss: {history['val_g_loss'][-1]:.4f} | "
                f"Physics (NMR: {history['nmr_loss'][-1]:.4f}, PE-Ca: {history['pe_ca_loss'][-1]:.4f})"
            )

        return history
