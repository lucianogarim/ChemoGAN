# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import gan_lib

# class ChemoGANTrainer:
#     def __init__(self, generator, discriminator, features, Z_dim):
#         self.generator = generator
#         self.discriminator = discriminator
#         self.features = features
#         self.Z = Z_dim

#         # Configurar Otimizadores
#         self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
#         self.g_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
#         self.d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)       

#         # Pesos e Hiperparâmetros
#         self.attr_weights = np.ones(len(features), dtype=np.float32)
#         self.SUAVIDADE = 1         
#         self.TV_ORDER = 2            
#         self.TV_HUBER_DELTA = None     
#         self.NMR_RESTRICAO = 10
#         self.NPHI_RHOB = 10
#         self.NPHI_DT = 10

#         # Mapeamento de índices 
#         try:
#             self.idx_nmre  = features.index('NMRE_FINAL')
#             self.idx_nmrfl = features.index('NMRFL_FINAL')
#             self.idx_nmrt  = features.index('NMRT_FINAL')
#             self.i_nphi = features.index('NPHI')
#             self.i_rhob = features.index('RHOB')
#             self.i_dt = features.index('DT')

#         except ValueError as e:
#             print(f"Aviso: Alguma feature necessária para penalidade não foi encontrada: {e}")
#             # Defina comportamento padrão ou pare o código se for crítico

#     @tf.function
#     def train_step(self, real_batch):
#         batch_size = tf.shape(real_batch)[0]
#         noise = tf.random.normal([batch_size, self.Z])

#         with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
#             fake_batch = self.generator(noise, training=True) 
            
#             # Discriminator
#             real_out = self.discriminator(real_batch, training=True)
#             fake_out = self.discriminator(fake_batch, training=True)

#             d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
#             d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)

            
#             # Penalidades (chamando do gan_lib)
#             tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
#             pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
#             pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
#             nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)

#             # Losses
#             d_loss = (d_loss_real + d_loss_fake)
#             g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)     

#             g_loss = (g_loss_adv 
#                       + self.SUAVIDADE * tv_term 
#                       + self.NMR_RESTRICAO * nmr_pen
#                       + self.NPHI_RHOB*pen_nphi_rhob
#                       + self.NPHI_DT*pen_nphi_dt)

#         g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
#         d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

#         self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
#         self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

#         return g_loss, d_loss, g_loss_adv


#     @tf.function
#     def val_step(self, val_batch):
#         batch_size = tf.shape(val_batch)[0]
#         noise = tf.random.normal([batch_size, self.Z])
#         fake_batch = self.generator(noise, training=False)

#         real_out = self.discriminator(val_batch, training=False)
#         fake_out = self.discriminator(fake_batch, training=False)

#         d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
#         d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)

#         # Penalidades (Reutilizando lógica)

#         tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
#         pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
#         pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
#         nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)

#         d_loss = (d_loss_real + d_loss_fake)

#         g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)

#         g_loss = (g_loss_adv 
#                   + self.SUAVIDADE * tv_term 
#                   + self.NMR_RESTRICAO * nmr_pen 
#                   + self.NPHI_RHOB*pen_nphi_rhob 
#                   + self.NPHI_DT*pen_nphi_dt)

#         return g_loss, d_loss



#     def fit(self, train_ds, val_ds, epochs):
#         print(f"Iniciando treino por {epochs} épocas...")
        
#         for epoch in range(1, epochs+1):
#             g_losses, d_losses = [], []
#             for real_batch in train_ds:
#                 gl, dl, gl_adv = self.train_step(real_batch)
#                 g_losses.append(gl)
#                 d_losses.append(dl)

#             val_g_losses, val_d_losses = [], []

#             for val_batch in val_ds:
#                 val_gl, val_dl = self.val_step(val_batch)
#                 val_g_losses.append(val_gl)
#                 val_d_losses.append(val_dl)

#             print(f"Epoch {epoch:03d} | "
#                   f"G_Loss: {tf.reduce_mean(g_losses):.4f} | D_Loss: {tf.reduce_mean(d_losses):.4f} | "
#                   f"Val G_Adv: {tf.reduce_mean(val_g_losses):.4f}")

#######################################
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import gan_lib

# class ChemoGANTrainer:
#     def __init__(self, generator, discriminator, features, Z_dim):
        
#         self.generator = generator
#         self.discriminator = discriminator
#         self.features = features
#         self.Z = Z_dim

#         # Configurar Otimizadores
#         self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
#         self.g_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
#         self.d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)        

#         # Pesos e Hiperparâmetros
#         self.attr_weights = np.ones(len(features), dtype=np.float32)
#         self.SUAVIDADE = 1         
#         self.TV_ORDER = 2            
#         self.TV_HUBER_DELTA = None   
#         self.NMR_RESTRICAO = 2
#         self.NPHI_RHOB = 2
#         self.NPHI_DT = 2

#         # Mapeamento de índices 
#         try:
#             self.idx_nmre  = features.index('NMRE_FINAL')
#             self.idx_nmrfl = features.index('NMRFL_FINAL')
#             self.idx_nmrt  = features.index('NMRT_FINAL')
#             self.i_nphi = features.index('NPHI')
#             self.i_rhob = features.index('RHOB')
#             self.idx_ca = features.index('DWCA')
#             self.i_dt = features.index('DT')
#             self.idx_pe = features.index('PE')
#         except ValueError as e:
#             print(f"Aviso: Alguma feature necessária para penalidade não foi encontrada: {e}")

#     @tf.function
#     def train_step(self, real_batch):
#         batch_size = tf.shape(real_batch)[0]
#         noise = tf.random.normal([batch_size, self.Z])

#         with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
#             fake_batch = self.generator(noise, training=True) 
            
#             # Discriminator
#             real_out = self.discriminator(real_batch, training=True)
#             fake_out = self.discriminator(fake_batch, training=True)

#             d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
#             d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
#             d_loss = (d_loss_real + d_loss_fake)
            
#             # Penalidades (chamando do gan_lib)
#             tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
#             pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
#             pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
#             # Penaliza se a correlação entre Cálcio e PE não for positiva
#             loss_pe_ca = gan_lib.corr_penalty(fake_batch[:, self.idx_pe], fake_batch[:, self.idx_ca], sign=1)

#             # Penaliza se a correlação entre Silício e PE não for negativa
#             #loss_pe_si = gan_lib.corr_penalty(fake_batch[:, idx_pe], fake_batch[:, idx_si], sign=-1)
            
#             nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)

#             # Generator Adversarial Loss
#             g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)     

#             g_loss = (g_loss_adv 
#                       + self.SUAVIDADE * tv_term 
#                       + self.NMR_RESTRICAO * nmr_pen
#                       + self.NPHI_RHOB * pen_nphi_rhob
#                       + self.NPHI_DT * pen_nphi_dt
#                       +loss_pe_ca)

#         g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
#         d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

#         self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
#         self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

#         # Retorna todas as parcelas da perda para monitoramento
#         return g_loss, d_loss, g_loss_adv, tv_term, nmr_pen, pen_nphi_rhob, pen_nphi_dt

#     @tf.function
#     def val_step(self, val_batch):
#         batch_size = tf.shape(val_batch)[0]
#         noise = tf.random.normal([batch_size, self.Z])
#         fake_batch = self.generator(noise, training=False)

#         real_out = self.discriminator(val_batch, training=False)
#         fake_out = self.discriminator(fake_batch, training=False)

#         d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
#         d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
#         d_loss = (d_loss_real + d_loss_fake)

#         # Penalidades
#         tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
#         pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
#         pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
#         nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)
        
#         loss_pe_ca = gan_lib.corr_penalty(fake_batch[:, self.idx_pe], fake_batch[:, self.idx_ca], sign=1)

#         g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)

#         g_loss = (g_loss_adv 
#                   + self.SUAVIDADE * tv_term 
#                   + self.NMR_RESTRICAO * nmr_pen 
#                   + self.NPHI_RHOB * pen_nphi_rhob 
#                   + self.NPHI_DT * pen_nphi_dt
#                   +loss_pe_ca)

#         return g_loss, d_loss

#     def fit(self, train_ds, val_ds, epochs):
#         print(f"Iniciando treino por {epochs} épocas...")
        
#         # Dicionário de Histórico para plote posterior
#         history = {
#             "g_loss": [], "d_loss": [], "g_adv": [],
#             "tv_loss": [], "nmr_loss": [], "corr_rhob_nphi": [], "corr_dt_nphi": [],
#             "val_g_loss": [], "val_d_loss": []
#         }
        
#         for epoch in range(1, epochs+1):
#             ep_g, ep_d, ep_adv = [], [], []
#             ep_tv, ep_nmr, ep_c1, ep_c2 = [], [], [], []
            
#             for real_batch in train_ds:
#                 gl, dl, adv, tv, nmr, c1, c2 = self.train_step(real_batch)
#                 ep_g.append(gl); ep_d.append(dl); ep_adv.append(adv)
#                 ep_tv.append(tv); ep_nmr.append(nmr); ep_c1.append(c1); ep_c2.append(c2)

#             val_g, val_d = [], []
#             for val_batch in val_ds:
#                 vgl, vdl = self.val_step(val_batch)
#                 val_g.append(vgl); val_d.append(vdl)

#             # Salvar estatísticas da época
#             history["g_loss"].append(np.mean(ep_g))
#             history["d_loss"].append(np.mean(ep_d))
#             history["g_adv"].append(np.mean(ep_adv))
#             history["tv_loss"].append(np.mean(ep_tv))
#             history["nmr_loss"].append(np.mean(ep_nmr))
#             history["corr_rhob_nphi"].append(np.mean(ep_c1))
#             history["corr_dt_nphi"].append(np.mean(ep_c2))
#             history["val_g_loss"].append(np.mean(val_g))
#             history["val_d_loss"].append(np.mean(val_d))

#             # Log limpo focado no que importa para P&D
#             print(f"Epoch {epoch:03d}/{epochs} | "
#                   f"G_Loss: {history['g_loss'][-1]:.4f} | D_Loss: {history['d_loss'][-1]:.4f} | "
#                   f"Val G_Loss: {history['val_g_loss'][-1]:.4f} | "
#                   f"Física (NMR: {history['nmr_loss'][-1]:.4f}, Corr: {history['corr_rhob_nphi'][-1]:.4f})")

#         return history


import tensorflow as tf
from tensorflow import keras
import numpy as np
import gan_lib

class ChemoGANTrainer:
    def __init__(self, generator, discriminator, features, Z_dim):
        
        self.generator = generator
        self.discriminator = discriminator
        self.features = features
        self.Z = Z_dim

        # Configurar Otimizadores
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
        self.d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)        

        # Pesos e Hiperparâmetros
        self.attr_weights = np.ones(len(features), dtype=np.float32)
        self.SUAVIDADE = 1         
        self.TV_ORDER = 2            
        self.TV_HUBER_DELTA = None   
        self.NMR_RESTRICAO = 2
        self.NPHI_RHOB = 2
        self.NPHI_DT = 2
        #self.PE_CA_WEIGHT = 2 # <--- NOVO: Peso para calibrar a correlação PE vs DWCA
        self.PE_SI_WEIGHT = 2
        # Mapeamento de índices 
        try:
            self.idx_nmre  = features.index('NMRE_FINAL')
            self.idx_nmrfl = features.index('NMRFL_FINAL')
            self.idx_nmrt  = features.index('NMRT_FINAL')
            self.i_nphi = features.index('NPHI')
            self.i_rhob = features.index('RHOB')
            #self.idx_ca = features.index('DWCA')
            self.i_dt = features.index('DT')
            self.idx_pe = features.index('PE')
            self.idx_si = features.index('DWSI')
        except ValueError as e:
            print(f"Aviso: Alguma feature necessária para penalidade não foi encontrada: {e}")

    
    def train_step(self, real_batch):
        batch_size = tf.shape(real_batch)[0]
        noise = tf.random.normal([batch_size, self.Z])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_batch = self.generator(noise, training=True) 
            
            # Discriminator
            real_out = self.discriminator(real_batch, training=True)
            fake_out = self.discriminator(fake_batch, training=True)

            d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
            d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
            d_loss = (d_loss_real + d_loss_fake)
            
            # Penalidades (chamando do gan_lib)
            tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
            pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
            pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
            nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)
            
            # Penaliza se a correlação entre Cálcio e PE não for positiva
            #loss_pe_ca = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_ca], sign=+1.0)
            # Penaliza se a correlação entre Cálcio e PE não for positiva
            loss_pe_si = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_si], sign=-1.0)

            # Generator Adversarial Loss
            g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)     

            g_loss = (g_loss_adv 
                      + self.SUAVIDADE * tv_term 
                      + self.NMR_RESTRICAO * nmr_pen
                      + self.NPHI_RHOB * pen_nphi_rhob
                      + self.NPHI_DT * pen_nphi_dt
                      #+ self.PE_CA_WEIGHT * loss_pe_ca)
                      + self.PE_SI_WEIGHT * loss_pe_si) # <--- Multiplicado pelo peso

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        
        return g_loss, d_loss, g_loss_adv, tv_term, nmr_pen, pen_nphi_rhob, pen_nphi_dt, loss_pe_si

    @tf.function
    def val_step(self, val_batch):
        batch_size = tf.shape(val_batch)[0]
        noise = tf.random.normal([batch_size, self.Z])
        fake_batch = self.generator(noise, training=False)

        real_out = self.discriminator(val_batch, training=False)
        fake_out = self.discriminator(fake_batch, training=False)

        d_loss_real = self.cross_entropy(tf.ones_like(real_out), real_out)
        d_loss_fake = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        d_loss = (d_loss_real + d_loss_fake)

        # Penalidades
        tv_term = gan_lib.tv_penalty_btA(fake_batch, self.attr_weights, self.TV_ORDER, self.TV_HUBER_DELTA)
        pen_nphi_rhob = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_rhob], sign=-1.0)
        pen_nphi_dt   = gan_lib.corr_penalty(fake_batch[..., self.i_nphi], fake_batch[..., self.i_dt],   sign=+1.0)
        nmr_pen = gan_lib.nmr_chain_penalty(fake_batch, self.idx_nmrt, self.idx_nmre, self.idx_nmrfl)
        
        #loss_pe_ca = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_ca], sign=+1.0)
        loss_pe_si = gan_lib.corr_penalty(fake_batch[..., self.idx_pe], fake_batch[..., self.idx_si], sign=-1.0)

        g_loss_adv = self.cross_entropy(tf.ones_like(fake_out), fake_out)

        g_loss = (g_loss_adv 
                  + self.SUAVIDADE * tv_term 
                  + self.NMR_RESTRICAO * nmr_pen 
                  + self.NPHI_RHOB * pen_nphi_rhob 
                  + self.NPHI_DT * pen_nphi_dt
                  #+ self.PE_CA_WEIGHT * loss_pe_ca)
                  + self.PE_SI_WEIGHT * loss_pe_si)

        return g_loss, d_loss

    def fit(self, train_ds, val_ds, epochs):
        print(f"Iniciando treino por {epochs} épocas...")
        
        # Dicionário de Histórico atualizado com pe_ca_loss
        history = {
            "g_loss": [], "d_loss": [], "g_adv": [],
            "tv_loss": [], "nmr_loss": [], "corr_rhob_nphi": [], "corr_dt_nphi": [],
            "pe_ca_loss": [], # <--- NOVO
            "val_g_loss": [], "val_d_loss": []
        }
        
        for epoch in range(1, epochs+1):
            ep_g, ep_d, ep_adv = [], [], []
            ep_tv, ep_nmr, ep_c1, ep_c2, ep_pe_ca = [], [], [], [], [] # <--- Atualizado
            
            for real_batch in train_ds:
                # Desempacotando o novo retorno (pe_ca)
                gl, dl, adv, tv, nmr, c1, c2, pe_ca = self.train_step(real_batch)
                
                ep_g.append(gl); ep_d.append(dl); ep_adv.append(adv)
                ep_tv.append(tv); ep_nmr.append(nmr); ep_c1.append(c1); ep_c2.append(c2)
                ep_pe_ca.append(pe_ca) # <--- NOVO

            val_g, val_d = [], []
            for val_batch in val_ds:
                vgl, vdl = self.val_step(val_batch)
                val_g.append(vgl); val_d.append(vdl)

            # Salvar estatísticas da época
            history["g_loss"].append(np.mean(ep_g))
            history["d_loss"].append(np.mean(ep_d))
            history["g_adv"].append(np.mean(ep_adv))
            history["tv_loss"].append(np.mean(ep_tv))
            history["nmr_loss"].append(np.mean(ep_nmr))
            history["corr_rhob_nphi"].append(np.mean(ep_c1))
            history["corr_dt_nphi"].append(np.mean(ep_c2))
            history["pe_ca_loss"].append(np.mean(ep_pe_ca)) # <--- NOVO
            history["val_g_loss"].append(np.mean(val_g))
            history["val_d_loss"].append(np.mean(val_d))

            # Log limpo focado no que importa (adicionei a loss do CA aqui para você ver na tela)
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"G_Loss: {history['g_loss'][-1]:.4f} | D_Loss: {history['d_loss'][-1]:.4f} | "
                  f"Val G_Loss: {history['val_g_loss'][-1]:.4f} | "
                  f"Física (NMR: {history['nmr_loss'][-1]:.4f}, PE-Ca: {history['pe_ca_loss'][-1]:.4f})")

        return history