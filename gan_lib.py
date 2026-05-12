import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================
# 1) Physics-Informed Penalty Functions
# ================================================


def tv_penalty_btA(y, attr_weights, order, huber_delta):
    """
    Compute a total-variation-style smoothness penalty over the depth/time axis.

    Parameters
    ----------
    y : tf.Tensor
        Generated or real sequence tensor with shape (batch, T, A), where T is
        the sequence length and A is the number of attributes/logs.
    attr_weights : array-like or None
        Optional attribute-wise weights. If provided, it must have length A.
    order : int
        Difference order used in the penalty. Must be 1 or 2.
        - order=1 penalizes first-order differences.
        - order=2 penalizes second-order differences.
    huber_delta : float or None
        If provided, applies a Huber-style penalty to the differences.
        If None, applies an absolute-value penalty.

    Returns
    -------
    tf.Tensor
        Scalar penalty value.
    """
    y = tf.cast(y, tf.float32)

    if order == 1:
        diffs = y[:, 1:, :] - y[:, :-1, :]
    elif order == 2:
        diffs = y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]
    else:
        raise ValueError("order must be 1 or 2")

    if huber_delta is None:
        pen = tf.abs(diffs)
    else:
        abs_d = tf.abs(diffs)
        quad = 0.5 * tf.square(abs_d)
        lin = huber_delta * (abs_d - 0.5 * huber_delta)
        pen = tf.where(abs_d <= huber_delta, quad, lin)

    if attr_weights is not None:
        w = tf.reshape(tf.cast(attr_weights, tf.float32), [1, 1, -1])
        pen = pen * w

    return tf.reduce_mean(pen)


def nmr_chain_penalty(y_scaled, i_rt, i_re, i_ff):
    """
    Penalize violations of the NMR porosity chain: RT >= RE >= FFL.

    This implementation locally reverses MinMax scaling using hardcoded scaler
    constants. The physical constraint is then applied in the original porosity
    space through a hinge-loss formulation.

    Parameters
    ----------
    y_scaled : tf.Tensor
        Scaled tensor with shape (..., A), where A contains the NMR attributes.
    i_rt : int
        Index of the RT/NMRT attribute.
    i_re : int
        Index of the RE/NMRE attribute.
    i_ff : int
        Index of the FFL/NMRFL attribute.

    Returns
    -------
    tf.Tensor
        Scalar penalty value for NMR hierarchy violations.
    """
    # 1. Real scaler values used to locally reverse MinMax scaling.
    rt_min = tf.constant(1.7000e-02, dtype=tf.float32)
    rt_max = tf.constant(2.99000e-01, dtype=tf.float32)
    re_min = tf.constant(6.0000e-03, dtype=tf.float32)
    re_max = tf.constant(2.89000e-01, dtype=tf.float32)
    ff_min = tf.constant(-3.6000e-02, dtype=tf.float32)
    ff_max = tf.constant(2.39000e-01, dtype=tf.float32)

    # 2. Locally undo MinMax scaling: X_raw = X_scaled * (max - min) + min.
    rt_raw = y_scaled[..., i_rt] * (rt_max - rt_min) + rt_min
    re_raw = y_scaled[..., i_re] * (re_max - re_min) + re_min
    ff_raw = y_scaled[..., i_ff] * (ff_max - ff_min) + ff_min

    # 3. Apply the physical constraint in real porosity space using hinge loss.
    p1 = tf.nn.relu(re_raw - rt_raw)  # Penalizes RE > RT.
    p2 = tf.nn.relu(ff_raw - re_raw)  # Penalizes FFL > RE.

    return tf.reduce_mean(p1 + p2)


def corr_penalty(a, b, sign):
    """
    Compute a local, window-wise Pearson-correlation penalty.

    Parameters
    ----------
    a : tf.Tensor
        First variable with shape (batch, T).
    b : tf.Tensor
        Second variable with shape (batch, T).
    sign : int or float
        Expected correlation sign.
        - If sign > 0, penalizes correlations below the positive threshold.
        - Otherwise, penalizes correlations above the negative threshold.

    Returns
    -------
    tf.Tensor
        Scalar correlation penalty.
    """
    # Center the values inside each window.
    a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
    b_mean = tf.reduce_mean(b, axis=1, keepdims=True)

    a_cent = a - a_mean
    b_cent = b - b_mean

    # Window-wise covariance and variance terms.
    num = tf.reduce_sum(a_cent * b_cent, axis=1)
    den = tf.sqrt(tf.reduce_sum(a_cent**2, axis=1) * tf.reduce_sum(b_cent**2, axis=1) + 1e-8)

    # r has shape (batch,), with one Pearson coefficient per sequence window.
    r = num / (den + 1e-8)
    threshold = 0.5

    if sign > 0:
        loss = tf.nn.relu(threshold - r)
    else:
        loss = tf.nn.relu(r - (-threshold))

    return tf.reduce_mean(loss)


def make_generator_1d_constrained(T, A, Z):
    """
    Build the constrained 1D generator model.

    The generator maps a latent vector of size Z into a sequence with shape
    (T, A). The convolutional stack uses progressively smaller kernels to capture
    large-scale geological trends, intermediate layer transitions, and finer
    carbonate heterogeneity.

    Parameters
    ----------
    T : int
        Sequence length.
    A : int
        Number of attributes/logs generated at each sequence step.
    Z : int
        Latent-space dimension.

    Returns
    -------
    keras.Sequential
        Generator model.
    """
    model = keras.Sequential(name="G1D_constrained")
    model.add(layers.Input(shape=(Z,)))

    # Initial dense projection.
    model.add(layers.Dense(T * 128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((T, 128)))

    # 1. Long vertical/asymmetric filter: captures broader geological trends.
    # Kernel size 11 lets the network observe a larger part of the profile at once.
    model.add(layers.Conv1D(128, kernel_size=11, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # 2. Medium filter: refines transitions between layers.
    model.add(layers.Conv1D(64, kernel_size=7, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # 3. Fine refinement filter: captures noisy carbonate heterogeneity.
    model.add(layers.Conv1D(32, kernel_size=3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # Output layer. kernel_size=1 correlates the generated tools/attributes (A).
    model.add(layers.Conv1D(A, kernel_size=1, padding="same", activation="sigmoid"))

    return model


def make_discriminator_1d_robust(T, A):
    """
    Build the robust 1D discriminator model.

    The discriminator receives sequences with shape (T, A) and outputs a single
    logit indicating whether the input is real or synthetic.

    Parameters
    ----------
    T : int
        Sequence length.
    A : int
        Number of attributes/logs per sequence step.

    Returns
    -------
    keras.Sequential
        Discriminator model.
    """
    model = keras.Sequential(name="D1D_Robust")
    model.add(layers.Input(shape=(T, A)))

    # Block 1
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv1D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    # Block 4
    model.add(layers.Conv1D(512, kernel_size=3, strides=1, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(1))

    return model
