import tensorflow as tf
from tensorflow import keras
import numpy as np
import diffusion_lib
from tqdm import tqdm

class DiffusionManager:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Parâmetros da Difusão (Schedule Linear)
        self.beta = np.linspace(beta_start, beta_end, timesteps).astype(np.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        
        # Constantes convertidas para Tensor float32 para evitar erro de tipo
        self.sqrt_alpha_bar = tf.constant(np.sqrt(self.alpha_bar), dtype=tf.float32)
        self.sqrt_one_minus_alpha_bar = tf.constant(np.sqrt(1.0 - self.alpha_bar), dtype=tf.float32)

    @tf.function
    def train_step(self, x_batch, optimizer):
        # Casting explícito da entrada para float32
        x_batch = tf.cast(x_batch, dtype=tf.float32)
        batch_size = tf.shape(x_batch)[0]
        
        # 1. Amostrar timesteps
        t = tf.random.uniform((batch_size, 1), minval=0, maxval=self.timesteps, dtype=tf.int32)
        
        # 2. Gerar ruído (dtype=float32)
        noise = tf.random.normal(tf.shape(x_batch), dtype=tf.float32)
        
        # 3. Obter parâmetros alpha para o tempo t
        sqrt_ab_t = tf.gather(self.sqrt_alpha_bar, t)[:, 0]
        sqrt_ab_t = tf.reshape(sqrt_ab_t, [batch_size, 1, 1])
        
        sqrt_one_minus_ab_t = tf.gather(self.sqrt_one_minus_alpha_bar, t)[:, 0]
        sqrt_one_minus_ab_t = tf.reshape(sqrt_one_minus_ab_t, [batch_size, 1, 1])
        
        # 4. Criar imagem ruidosa
        x_noisy = sqrt_ab_t * x_batch + sqrt_one_minus_ab_t * noise
        
        with tf.GradientTape() as tape:
            # 5. Prever o ruído
            noise_pred = self.model([x_noisy, tf.cast(t, tf.float32)], training=True)
            
            # 6. Loss MSE (Cálculo Manual para evitar AttributeError do Keras)
            # Diferença -> Quadrado -> Média Global
            loss = tf.reduce_mean(tf.square(noise - noise_pred))
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss

    def generate(self, num_samples, T, A):
        """
        Executa o processo reverso (denoising).
        """
        print(f"Gerando {num_samples} amostras via Difusão Reversa...")
        
        # Iniciar com ruído puro float32
        x = tf.random.normal((num_samples, T, A), dtype=tf.float32)
        
        # Loop reverso
        for i in tqdm(reversed(range(0, self.timesteps))):
            t_vec = tf.ones((num_samples, 1), dtype=tf.float32) * i
            
            # Prever o ruído
            pred_noise = self.model([x, t_vec], training=False)
            
            # Pegar escalares do numpy
            alpha_t = self.alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.beta[i]
            
            # Langevin Dynamics
            if i > 0:
                noise = tf.random.normal(tf.shape(x), dtype=tf.float32)
            else:
                noise = 0.0
                
            # Coeficientes matemáticos
            term1 = 1 / np.sqrt(alpha_t)
            term2 = (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * pred_noise)
            sigma = np.sqrt(beta_t)
            
            x = term1 * term2 + sigma * noise
            
        return x.numpy()

def train_diffusion(X_train, T, A, epochs=100, batch_size=64):
    # Instanciar Modelo
    unet = diffusion_lib.make_unet_1d(T, A)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    
    # Instanciar Gerenciador
    manager = DiffusionManager(unet)
    
    # Dataset
    train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(4096).batch(batch_size)
    
    print(f"\n===== Iniciando Treino do Diffusion Model ({epochs} épocas) =====")
    
    for epoch in range(epochs):
        loss_mean = keras.metrics.Mean()
        
        for batch in train_ds:
            loss = manager.train_step(batch, optimizer)
            loss_mean.update_state(loss)
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss_mean.result():.5f}")
            
    return manager