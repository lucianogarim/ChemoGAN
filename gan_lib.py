# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # ================================================
# # 1) Funções de Penalidade
# # ================================================

# def tv_penalty_btA(y, attr_weights, order, huber_delta):
#     y = tf.cast(y, tf.float32)
#     if order == 1:
#         diffs = y[:, 1:, :] - y[:, :-1, :]
#     elif order == 2:
#         diffs = (y[:, 2:, :] - 2.0*y[:, 1:-1, :] + y[:, :-2, :])
#     else:
#         raise ValueError("order deve ser 1 ou 2")


#     if huber_delta is None:
#         pen = tf.abs(diffs)
#     else:
#         abs_d = tf.abs(diffs)
#         quad = 0.5 * tf.square(abs_d)
#         lin  = huber_delta * (abs_d - 0.5*huber_delta)
#         pen = tf.where(abs_d <= huber_delta, quad, lin)
        
#     if attr_weights is not None:
#         w = tf.reshape(tf.cast(attr_weights, tf.float32), [1, 1, -1])
#         pen = pen * w

#     return tf.reduce_mean(pen)


# def nmr_chain_penalty(y, i_rt, i_re, i_ff):
#     p1 = tf.nn.relu(y[..., i_re] - y[..., i_rt])  # RE <= RT
#     p2 = tf.nn.relu(y[..., i_ff] - y[..., i_re])  # FFL <= RE

#     return tf.reduce_mean(p1 + p2)



# def corr_penalty(a, b, sign):
#     a = a - tf.reduce_mean(a)
#     b = b - tf.reduce_mean(b)
#     num = tf.reduce_mean(a*b)
#     den = tf.sqrt(tf.reduce_mean(a*a)*tf.reduce_mean(b*b) + 1e-8)
#     r = num / (den + 1e-8)
    
#     threshold = 0.7

#     if sign > 0:
#         # Queremos r > 0.7. Puna se r < 0.7
#         loss = tf.nn.relu(threshold - r)
#     else:
#         # Queremos r < -0.7. Puna se r > -0.7
#         # Ex: r = -0.9 -> -0.9 - (-0.7) = -0.2 (ReLU=0) OK
#         # Ex: r = 0.0  ->  0.0 - (-0.7) = +0.7 (ReLU=0.7) PUNIÇÃO OK
#         loss = tf.nn.relu(r - (-threshold))

#     return tf.reduce_mean(loss)

# #================================================
# #2) Definição dos Modelos
# #================================================
# def make_generator_1d_constrained(T, A, Z):
#     model = keras.Sequential(name="G1D_constrained")
#     model.add(layers.Input(shape=(Z,)))
#     model.add(layers.Dense(T * 128, use_bias=False)) 
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Reshape((T, 128))) 
#     model.add(layers.Conv1D(128, kernel_size=5, padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Conv1D(64, kernel_size=5, padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Conv1D(A, kernel_size=1, padding='same', activation='sigmoid')) 

#     return model


# def make_discriminator_1d_robust(T, A):
#     model = keras.Sequential(name="D1D_Robust")
#     model.add(layers.Input(shape=(T, A)))

#     # Bloco 1
#     model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding='same'))
#     model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

#     # Bloco 2
#     model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding='same'))
#     model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

#     # Bloco 3
#     model.add(layers.Conv1D(256, kernel_size=5, strides=2, padding='same'))
#     model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

#     # Bloco 4
#     model.add(layers.Conv1D(512, kernel_size=3, strides=1, padding='same'))
#     model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

#     model.add(layers.GlobalAveragePooling1D())
#     model.add(layers.Dense(1))

#     return model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================
# 1) Funções de Penalidade (Physics-Informed)
# ================================================

def tv_penalty_btA(y, attr_weights, order, huber_delta):
    y = tf.cast(y, tf.float32)
    if order == 1:
        diffs = y[:, 1:, :] - y[:, :-1, :]
    elif order == 2:
        diffs = (y[:, 2:, :] - 2.0*y[:, 1:-1, :] + y[:, :-2, :])
    else:
        raise ValueError("order deve ser 1 ou 2")

    if huber_delta is None:
        pen = tf.abs(diffs)
    else:
        abs_d = tf.abs(diffs)
        quad = 0.5 * tf.square(abs_d)
        lin  = huber_delta * (abs_d - 0.5*huber_delta)
        pen = tf.where(abs_d <= huber_delta, quad, lin)
        
    if attr_weights is not None:
        w = tf.reshape(tf.cast(attr_weights, tf.float32), [1, 1, -1])
        pen = pen * w

    return tf.reduce_mean(pen)

def nmr_chain_penalty(y_scaled, i_rt, i_re, i_ff):
    """
    Penaliza violações na cadeia de porosidade do RMN: RT >= RE >= FFL.
    Solução Local: Desnormalização com constantes hardcoded.
    """
    # 1. INSIRA AQUI OS VALORES REAIS DO SEU SCALER 
    rt_min, rt_max = tf.constant(1.7000e-02, dtype=tf.float32), tf.constant(2.99000e-01, dtype=tf.float32)
    re_min, re_max = tf.constant(6.0000e-03, dtype=tf.float32), tf.constant(2.89000e-01, dtype=tf.float32)
    ff_min, ff_max = tf.constant(-3.6000e-02, dtype=tf.float32), tf.constant(2.39000e-01, dtype=tf.float32)
    
    # 2. Desfazer o MinMaxScaler localmente: X_raw = X_scaled * (max - min) + min
    rt_raw = y_scaled[..., i_rt] * (rt_max - rt_min) + rt_min
    re_raw = y_scaled[..., i_re] * (re_max - re_min) + re_min
    ff_raw = y_scaled[..., i_ff] * (ff_max - ff_min) + ff_min
    
    # 3. Aplicar a restrição física na porosidade real (Hinge Loss)
    p1 = tf.nn.relu(re_raw - rt_raw)  # Penaliza se RE > RT
    p2 = tf.nn.relu(ff_raw - re_raw)  # Penaliza se FFL > RE
    
    return tf.reduce_mean(p1 + p2)



def corr_penalty(a, b, sign):
    """
    Calcula a correlação de Pearson de forma local (window-wise).
    a e b possuem shape: (Batch, T)
    """
    # Centraliza os dados na janela
    a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
    b_mean = tf.reduce_mean(b, axis=1, keepdims=True)
    
    a_cent = a - a_mean
    b_cent = b - b_mean
    
    # Covariância e Variância por janela
    num = tf.reduce_sum(a_cent * b_cent, axis=1)
    den = tf.sqrt(tf.reduce_sum(a_cent**2, axis=1) * tf.reduce_sum(b_cent**2, axis=1) + 1e-8)
    
    # r tem shape (Batch,) - Um coeficiente de Pearson para cada amostra da janela
    r = num / (den + 1e-8)
    threshold = 0.5

    if sign > 0:
        # Queremos r > 0.7. Puna se r < 0.7
        loss = tf.nn.relu(threshold - r)
        #loss = tf.math.softplus(threshold - r) #ok
        #loss = tf.math.exp(threshold - r)
    else:
        # Queremos r < -0.7. Puna se r > -0.7
        loss = tf.nn.relu(r - (-threshold))
        #loss = tf.math.softplus(r - (-threshold)) #ok
        #loss = tf.math.exp(r - (-threshold))
        

    return tf.reduce_mean(loss)

# ================================================
# 2) Definição das Arquiteturas
# ================================================

# def make_generator_1d_constrained(T, A, Z):
#     model = keras.Sequential(name="G1D_constrained")
#     model.add(layers.Input(shape=(Z,)))
#     model.add(layers.Dense(T * 128, use_bias=False)) 
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Reshape((T, 128))) 
    
#     model.add(layers.Conv1D(128, kernel_size=20, padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
    
#     model.add(layers.Conv1D(64, kernel_size=10, padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU(0.2))
    
#     model.add(layers.Conv1D(A, kernel_size=1, padding='same', activation='sigmoid')) 
#     return model

def make_generator_1d_constrained(T, A, Z):
    model = keras.Sequential(name="G1D_constrained")
    model.add(layers.Input(shape=(Z,)))
    
    # Camada densa inicial
    model.add(layers.Dense(T * 128, use_bias=False)) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((T, 128))) 
    
    # 1. Filtro Longo (Assimétrico Vertical): Captura tendências geológicas maiores
    # Em vez de 5, usamos 11 para "enxergar" uma porção maior do perfil de uma vez
    model.add(layers.Conv1D(128, kernel_size=11, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    # 2. Filtro Médio: Refina as transições entre camadas
    model.add(layers.Conv1D(64, kernel_size=7, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    # 3. Filtro de Refinamento Fino: Captura a heterogeneidade ruidosa do carbonato
    model.add(layers.Conv1D(32, kernel_size=3, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    # Camada de Saída: kernel_size=1 está correto para correlacionar as ferramentas (A)
    model.add(layers.Conv1D(A, kernel_size=1, padding='same', activation='sigmoid')) 
    return model



def make_discriminator_1d_robust(T, A):
    model = keras.Sequential(name="D1D_Robust")
    model.add(layers.Input(shape=(T, A)))

    # Bloco 1
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

    # Bloco 2
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

    # Bloco 3
    model.add(layers.Conv1D(256, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

    # Bloco 4
    model.add(layers.Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(1))
    return model


