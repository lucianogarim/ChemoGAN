import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================
# 1) Blocos Construtivos do Transformer
# ================================================

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) para amostrar z (Reparameterization Trick)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ================================================
# 2) Definição de Encoder e Decoder
# ================================================

def make_transformer_encoder(T, A, embed_dim, latent_dim):
    inputs = layers.Input(shape=(T, A))
    
    # Projeção inicial (Dense) para atingir dimensão de embedding
    x = layers.Dense(embed_dim)(inputs)
    
    # Positional Embedding
    positions = tf.range(start=0, limit=T, delta=1)
    pos_emb = layers.Embedding(input_dim=T, output_dim=embed_dim)(positions)
    x = x + pos_emb
    
    # Blocos Transformer
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    
    # Global Average Pooling para condensar o tempo
    x = layers.GlobalAveragePooling1D()(x)
    
    # Camadas Latentes
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def make_transformer_generator(T, A, latent_dim, embed_dim):
    """
    Decoder que atua como Gerador.
    Entrada: Ruído (Z)
    Saída: Sequência (T, A)
    """
    inputs = layers.Input(shape=(latent_dim,))
    
    # Mapeia Z para a dimensão da sequência (T * embed_dim)
    x = layers.Dense(T * embed_dim)(inputs)
    x = layers.Reshape((T, embed_dim))(x)
    
    # Adiciona Positional Encoding novamente
    positions = tf.range(start=0, limit=T, delta=1)
    pos_emb = layers.Embedding(input_dim=T, output_dim=embed_dim)(positions)
    x = x + pos_emb
    
    # Blocos Transformer
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(embed_dim, num_heads=4, ff_dim=128)(x)
    
    # Saída final sigmoid (dados normalizados MinMax 0-1)
    outputs = layers.Dense(A, activation="sigmoid")(x)
    
    decoder = keras.Model(inputs, outputs, name="generator_transformer")
    return decoder

# ================================================
# 3) Classe VAE Principal
# ================================================

class TransformerVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # --- CORREÇÃO: Cálculo manual do MSE para evitar erro de atributo ---
            # 1. Diferença quadrática
            squared_diff = tf.square(data - reconstruction)
            # 2. Média através das features (axis=-1) -> Retorna (Batch, T)
            mse_per_timestep = tf.reduce_mean(squared_diff, axis=-1)
            # 3. Soma através do tempo (axis=1) -> Retorna (Batch,)
            sum_mse_time = tf.reduce_sum(mse_per_timestep, axis=1)
            # 4. Média do batch (Escalar)
            reconstruction_loss = tf.reduce_mean(sum_mse_time)
            # -------------------------------------------------------------------
            
            # Loss KL (Divergência de Kullback-Leibler)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Peso do termo KL (beta). 0.1 ajuda a priorizar a reconstrução (qualidade visual)
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