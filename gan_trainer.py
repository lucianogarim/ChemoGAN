"""ChemoGAN trainer (revision 1).

Changes w.r.t. the first submission:
* constraint set (pairs, tau, NMR scaling) comes from physics.ConstraintSet, i.e. the
  same definition used by the PVR and the rejection filter; the target-specific pair
  (PE-DWSI or PE-DWCA) is selected from config instead of being commented in/out;
* lambdas can be set to zero to obtain the unconstrained GAN used in the ablation;
* train_step is compiled with tf.function and all random generators are seeded;
* optional one-sided label smoothing (off by default).
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ChemoGANTrainer:
    def __init__(self, generator, discriminator, constraints, Z_dim, lambdas=None,
                 lr=1e-4, label_smoothing=0.0):
        self.G = generator
        self.D = discriminator
        self.C = constraints
        self.Z = Z_dim
        lam = {'tv': 1.0, 'nmr': 2.0, 'corr': 2.0} if lambdas is None else lambdas
        self.l_tv = float(lam.get('tv', 0.0))
        self.l_nmr = float(lam.get('nmr', 0.0))
        self.l_corr = float(lam.get('corr', 0.0))
        self.smooth = float(label_smoothing)
        self.ce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        self.d_opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        self.n_pairs = len(self.C.i_pairs)

    def _penalties(self, fake):
        y = tf.cast(fake, tf.float32)
        d2 = y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]
        tv = tf.reduce_mean(tf.abs(d2))
        nmr = self.C.tf_nmr_penalty(y)
        corr = self.C.tf_corr_penalties(y)
        return tv, nmr, corr

    @tf.function
    def train_step(self, real):
        b = tf.shape(real)[0]
        noise = tf.random.normal([b, self.Z])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = self.G(noise, training=True)
            ro = self.D(real, training=True)
            fo = self.D(fake, training=True)
            d_loss = self.ce(tf.ones_like(ro) * (1.0 - self.smooth), ro) + self.ce(tf.zeros_like(fo), fo)
            adv = self.ce(tf.ones_like(fo), fo)
            tv, nmr, corr = self._penalties(fake)
            corr_sum = tf.add_n(corr) if corr else tf.constant(0.0)
            g_loss = adv + self.l_tv * tv + self.l_nmr * nmr + self.l_corr * corr_sum
        self.g_opt.apply_gradients(zip(gt.gradient(g_loss, self.G.trainable_variables), self.G.trainable_variables))
        self.d_opt.apply_gradients(zip(dt.gradient(d_loss, self.D.trainable_variables), self.D.trainable_variables))
        return tf.stack([g_loss, d_loss, adv, tv, nmr] + list(corr))

    @tf.function
    def val_step(self, real):
        b = tf.shape(real)[0]
        fake = self.G(tf.random.normal([b, self.Z]), training=False)
        ro = self.D(real, training=False)
        fo = self.D(fake, training=False)
        d_loss = self.ce(tf.ones_like(ro), ro) + self.ce(tf.zeros_like(fo), fo)
        adv = self.ce(tf.ones_like(fo), fo)
        return tf.stack([adv, d_loss])

    def fit(self, train_ds, val_ds, epochs, snapshot_epochs=(), snapshot_fn=None, log_every=50):
        keys = ['g_loss', 'd_loss', 'g_adv', 'tv', 'nmr'] + [f'corr_{a}_{b}' for a, b, _ in self.C.pairs]
        hist = {k: [] for k in keys + ['val_g_adv', 'val_d_loss', 'sec']}
        t0 = time.time()
        for ep in range(1, epochs + 1):
            vals = np.mean([self.train_step(x).numpy() for x in train_ds], axis=0)
            for k, v in zip(keys, vals):
                hist[k].append(float(v))
            if val_ds is not None:
                vv = np.mean([self.val_step(x).numpy() for x in val_ds], axis=0)
                hist['val_g_adv'].append(float(vv[0])); hist['val_d_loss'].append(float(vv[1]))
            hist['sec'].append(time.time() - t0)
            if ep in snapshot_epochs and snapshot_fn is not None:
                snapshot_fn(ep, self.G)
            if ep % log_every == 0 or ep == 1:
                print(f"  ep {ep:4d} | G {hist['g_loss'][-1]:.3f} D {hist['d_loss'][-1]:.3f} "
                      f"adv {hist['g_adv'][-1]:.3f} nmr {hist['nmr'][-1]:.2e} | {hist['sec'][-1]:.0f}s", flush=True)
        return hist
