# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:43:32 2026

@author: LUCIANOGARIM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

# ================================================
# 1) Blocos Construtivos (Time Embedding & ResBlock)
# ================================================

class SinusoidalTimeEmbedding(layers.Layer):
    """Codifica o passo de tempo t em um vetor denso."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb_scale = math.log(10000) / (self.half_dim - 1)

    def call(self, time):
        # time shape: (batch_size, 1)
        emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb_scale)
        emb = time * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class ResidualBlock1D(layers.Layer):
    """Bloco Convolucional que mistura features com o Time Embedding."""
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding="same")
        self.conv2 = layers.Conv1D(filters, kernel_size, padding="same")
        self.norm1 = layers.GroupNormalization(groups=8) # GN é melhor que BN para Diffusion
        self.norm2 = layers.GroupNormalization(groups=8)
        self.act = layers.Activation("swish") # Swish/SiLU é padrão em Diffusion
        self.time_proj = layers.Dense(filters, activation="swish") # Projeta tempo p/ filtros
        self.shortcut = layers.Conv1D(filters, 1, padding="same") 

    def call(self, x, time_emb):
        # x: (batch, steps, channels)
        # time_emb: (batch, time_dim)
        
        h = self.conv1(self.act(self.norm1(x)))
        
        # Adiciona embedding de tempo (broadcasting no eixo steps)
        # time_emb projected: (batch, filters) -> (batch, 1, filters)
        t_vec = self.time_proj(time_emb)
        h = h + t_vec[:, tf.newaxis, :]
        
        h = self.conv2(self.act(self.norm2(h)))
        
        if x.shape[-1] != h.shape[-1]:
            x = self.shortcut(x)
            
        return x + h

# ================================================
# 2) A U-Net 1D (O Modelo Denoiser)
# ================================================

def make_unet_1d(T, A, time_steps=1000):
    """
    U-Net adaptada para sequências curtas (T=20).
    Entrada: (Batch, T, A) + (Batch, 1) [tempo]
    Saída: (Batch, T, A) [ruído previsto]
    """
    
    # Entradas
    input_x = layers.Input(shape=(T, A))
    input_t = layers.Input(shape=(1,)) # Inteiro do timestep
    
    # 1. Processar Tempo
    time_dim = 64
    t_emb = SinusoidalTimeEmbedding(time_dim)(input_t)
    # MLP para processar o embedding cru
    t_emb = layers.Dense(time_dim * 4, activation="swish")(t_emb)
    t_emb = layers.Dense(time_dim)(t_emb)
    
    # 2. Encoder (Downsampling)
    # Como T=20 é pequeno, fazemos downsampling cuidadoso: 20 -> 10 -> 5
    
    # Initial Conv
    x = layers.Conv1D(64, 3, padding="same")(input_x)
    
    # Block 1 (Resolução 20)
    x1 = ResidualBlock1D(64)(x, t_emb)
    
    # Down 1 (20 -> 10)
    x = layers.Conv1D(128, 3, strides=2, padding="same")(x1)
    x2 = ResidualBlock1D(128)(x, t_emb)
    
    # Down 2 (10 -> 5)
    x = layers.Conv1D(256, 3, strides=2, padding="same")(x2)
    x3 = ResidualBlock1D(256)(x, t_emb)
    
    # 3. Bottleneck (Resolução 5)
    mid = ResidualBlock1D(256)(x3, t_emb)
    mid = ResidualBlock1D(256)(mid, t_emb)
    
    # 4. Decoder (Upsampling com Skip Connections)
    
    # Up 1 (5 -> 10)
    x = layers.UpSampling1D(size=2)(mid)
    x = layers.Concatenate()([x, x2])
    x = ResidualBlock1D(128)(x, t_emb)
    
    # Up 2 (10 -> 20)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Concatenate()([x, x1])
    x = ResidualBlock1D(64)(x, t_emb)
    
    # 5. Saída Final
    # Queremos prever o ruído, que tem a mesma dimensão da entrada (A)
    outputs = layers.Conv1D(A, 3, padding="same")(x)
    
    model = keras.Model(inputs=[input_x, input_t], outputs=outputs, name="unet_1d_diff")
    return model