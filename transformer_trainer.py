# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:22:24 2026

@author: LUCIANOGARIM
"""

import tensorflow as tf
from tensorflow import keras
import transformer_lib

def train_transformer(X_train, T, A, Z_dim, epochs=100, batch_size=64):
    """
    Configura e treina o Transformer VAE.
    Retorna o modelo 'Decoder' que serve como Gerador.
    """
    
    # Hiperparâmetros do Transformer
    embed_dim = 64  # Dimensão dos vetores de atenção
    
    # Instanciar redes
    encoder = transformer_lib.make_transformer_encoder(T, A, embed_dim, Z_dim)
    decoder = transformer_lib.make_transformer_generator(T, A, Z_dim, embed_dim)
    
    # Instanciar VAE
    vae = transformer_lib.TransformerVAE(encoder, decoder)
    
    # Compilar
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    
    print(f"\n===== Iniciando Treino do Transformer VAE ({epochs} épocas) =====")
    vae.fit(X_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    return decoder # Retornamos o decoder pois ele é o "Generator"

