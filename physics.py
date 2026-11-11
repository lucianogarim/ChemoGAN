"""Petrophysical constraint specification shared by training, PVR and rejection sampling.

Revision 1 changes (reviewers R1.4, R2.20, R2.24, R4.2, R4.3):
* the same constraint set and the same threshold tau are used in the loss, in the PVR
  and in the rejection filter (the first submission used tau=0.5 in training but 0.7 in
  the PVR, and silently skipped the NMR check because of a column-name mismatch);
* the PVR is decomposed into a hard component (NMR chain, a physical law) and a soft
  component (window-wise correlation priors);
* the same statistics are computed on REAL windows, giving a reference PVR that a
  perfect generator would reproduce.
"""
import numpy as np
import tensorflow as tf


class ConstraintSet:
    def __init__(self, features, pairs, tau, nmr_cols, nmr_tol, scaler, nmr_margin=0.0):
        self.features = list(features)
        self.pairs = [(a, b, s) for (a, b, s) in pairs]
        self.tau = float(tau)
        self.nmr_tol = float(nmr_tol)
        self.nmr_margin = float(nmr_margin)
        self.i_nmr = [self.features.index(c) for c in nmr_cols]  # T, E, FL
        self.i_pairs = [(self.features.index(a), self.features.index(b), s) for a, b, s in self.pairs]
        # MinMax parameters taken from the fitted scaler (no hard-coded constants).
        self.data_min = np.asarray(scaler.data_min_, dtype=np.float32)
        self.data_range = np.asarray(scaler.data_range_, dtype=np.float32)

    # ------------------------------------------------------------------ numpy (eval)
    def to_physical(self, X_scaled):
        return X_scaled * self.data_range + self.data_min

    @staticmethod
    def _window_corr(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
        with np.errstate(invalid='ignore', divide='ignore'):
            r = (a * b).sum(1) / den
        return r  # NaN when a window is constant

    def violations(self, X_phys):
        """X_phys: (N, T, A) in physical units. Returns dict of boolean (N,) masks."""
        it, ie, ifl = self.i_nmr
        nmr = ((X_phys[:, :, ie] > X_phys[:, :, it] + self.nmr_tol) |
               (X_phys[:, :, ifl] > X_phys[:, :, ie] + self.nmr_tol)).any(axis=1)
        out = {'nmr': nmr}
        soft = np.zeros(len(X_phys), dtype=bool)
        for (ia, ib, s), (a, b, _) in zip(self.i_pairs, self.pairs):
            r = self._window_corr(X_phys[:, :, ia], X_phys[:, :, ib])
            v = np.isnan(r) | ((r < self.tau) if s > 0 else (r > -self.tau))
            out[f'corr_{a}_{b}'] = v
            soft |= v
        out['soft'] = soft
        out['any'] = nmr | soft
        return out

    def pvr(self, X_phys):
        v = self.violations(X_phys)
        return {k: 100.0 * float(np.mean(m)) for k, m in v.items()}, v

    # ------------------------------------------------------------- tensorflow (loss)
    def tf_nmr_penalty(self, y):
        it, ie, ifl = self.i_nmr
        rng = tf.constant(self.data_range)
        mn = tf.constant(self.data_min)
        phys = y * rng + mn
        m = self.nmr_margin
        p1 = tf.nn.relu(phys[..., ie] - phys[..., it] + m)
        p2 = tf.nn.relu(phys[..., ifl] - phys[..., ie] + m)
        return tf.reduce_mean(p1 + p2)

    def tf_corr_penalty(self, y, ia, ib, sign):
        a = y[..., ia]
        b = y[..., ib]
        a = a - tf.reduce_mean(a, axis=1, keepdims=True)
        b = b - tf.reduce_mean(b, axis=1, keepdims=True)
        num = tf.reduce_sum(a * b, axis=1)
        den = tf.sqrt(tf.reduce_sum(a ** 2, axis=1) * tf.reduce_sum(b ** 2, axis=1) + 1e-8)
        r = num / (den + 1e-8)
        if sign > 0:
            return tf.reduce_mean(tf.nn.relu(self.tau - r))
        return tf.reduce_mean(tf.nn.relu(r + self.tau))

    def tf_corr_penalties(self, y):
        return [self.tf_corr_penalty(y, ia, ib, s) for ia, ib, s in self.i_pairs]
