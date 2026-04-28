import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.model_selection import train_test_split

# 1. Garante que o Windows ache as DLLs do Conda
conda_prefix = sys.prefix
bin_path = os.path.join(conda_prefix, 'Library', 'bin')

if bin_path not in os.environ['PATH']:

    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
# 2. Configura a memória para não dar erro de alocação
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Pronta: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(e)
# ---------------------------------------------------------------
# Imports dos seus módulos
import pre_processamento as process
import plots
import gan_lib
import xgb
from gan_trainer import ChemoGANTrainer
import transformer_trainer 
import diffusion_trainer

# ====================================================
# 1) Configuração e Carga de Dados
# ====================================================
data_dir = r'C:\Users\lucianogarim\Associacao Antonio Vieira\UNI - Projeto Quimio - General\Projeto Quimioestratigrafia\Dados\Poços_Sapinhoá'

mnemonicos = ['DEPTH','HCAL','DWAL', 'DWCA', 'DWFE', 'DWSI', 'RHOB', 'GR', 'NPHI', 'PE', 'DT',
              'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL', 'HFK', 'HTHO', 'HURA','T2LM']

filtros = {
    '7-SPH-3-SPS_BASE.las': (5037, 5450),
    '7-SPH-20D-SPS_BASE.las': (5050, 5277), '7-SPH-15D-SPS_BASE.las': (5430, 5722), '1-SPS-96-SP_BASE.las': (5037, 5483),
    '3-SPS-69-SP_BASE.las': (4964, 5175), '3-SPS-82A-SP_BASE.las': (5076,5494), '7-SPH-1-SPS_BASE.las': (4995,5357),
    '7-SPH-4D-SPS_BASE.las': (5150, 5400), '7-SPH-5-SPS_BASE.las': (5053, 5190),
    '7-SPH-6-SPS_BASE.las': (5000, 5368), '7-SPH-7D-SPS_BASE.las': (5220, 5487), '7-SPH-14D-SPS_BASE.las': (5193, 5460),
    '7-SPH-16D-SPS_BASE.las': (5145, 5285), '7-SPH-17-SPS_BASE.las': (5031, 5200), '7-SPH-22-SPS_BASE.las': (4980, 5030),
    '8-SPH-9-SPS_BASE.las': (5128, 5356), '8-SPH-11-SPS_BASE.las': (5091, 5280), 
    '8-SPH-21D-SPS_BASE.las': (5117,5496), '8-SPH-23-SPS_BASE.las': (5105, 5478), '9-SPS-77A-SP_BASE.las': (5023,5339),
    '9-SPS-97-SP_BASE.las' : (5138,5190)
}


lista_pocos_cegos = ['7-SPH-3-SPS_BASE.las', '9-SPS-77A-SP_BASE.las','9-SPS-97-SP_BASE.las']
features =  ['GR','PE','DT', 'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL', 'HFK', 'HTHO', 'HURA','T2LM','RHOB','NPHI','DWSI']


n_steps = 40

# Pipeline de Processamento
dicionario_pocos = process.carregamento(data_dir, mnemonicos)
dicionario_pocos = process.filtro_profundidade(dicionario_pocos, filtros)
dicionario_pocos = process.trata_outliers(dicionario_pocos, features)
dicionario_pocos, pocos_cegos = process.escolhe_pocos_cegos(dicionario_pocos, lista_pocos_cegos)


# Concatenamos apenas para TREINAR O SCALER
pocos_concatenados = process.concatena_pocos(dicionario_pocos)
_, scaler = process.muda_escala(pocos_concatenados, features)

# Criamos as sequências
X = process.cria_sequencias(dicionario_pocos, features, n_steps, scaler) 

# Params globais
T = X.shape[1] 
A = X.shape[2] 
Z = 100
BATCH_SIZE = 512


# ====================================================
# Funções Auxiliares de Métricas
# ====================================================
def dist_metrics_per_feature(real, synth, features):
    """
    Calcula métricas de similaridade de distribuição entre dados reais e sintéticos.
    """
    results = []
    
    for i, feat in enumerate(features):

        r = real[:, i]
        s = synth[:, i]     

        # Limpeza INDEPENDENTE de NaNs e Infinitos

        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]

        if len(r) == 0 or len(s) == 0:

            continue

        # Métricas de Distribuição
        wd = wasserstein_distance(r, s)
        ks_stat, _ = ks_2samp(r, s)

        mean_diff = abs(np.mean(r) - np.mean(s))
        std_diff  = abs(np.std(r) - np.std(s))  

        results.append({

            "Feature": feat,
            "wasserstein": wd,
            "ks_stat": ks_stat,
            "mean_diff": mean_diff,
            "std_diff": std_diff
        })
       
    return pd.DataFrame(results)


def corr_error(real, synth):
    """
    Calcula o MAE entre as matrizes de correlação.
    """
    c_real = np.corrcoef(real, rowvar=False)
    c_synth = np.corrcoef(synth, rowvar=False)

    c_real = np.nan_to_num(c_real, nan=0.0)
    c_synth = np.nan_to_num(c_synth, nan=0.0)

    diff = np.abs(c_real - c_synth)

    return np.mean(diff)

    
def calculate_pvr(synth_2d, features, n_steps, scaler=None):
    """
    Calcula a Taxa de Violação Física (PVR - Physics Violation Rate).
    Uma janela (amostra de tamanho T) é considerada inválida se:
    1. Violar a hierarquia do RMN (NMRT >= NMRE >= NMRFL) em qualquer ponto.
    2. Violar as correlações estruturais (RHOB vs NPHI < -0.7 ou NPHI vs DT > 0.7).
    """

    # --- NOVO: Desnormalização Global ---
    # Se um scaler for fornecido, voltamos o dado para o espaço físico real
    # onde a termodinâmica de fato funciona.
    if scaler is not None:
        # Cria uma cópia para não alterar o array original fora da função
        synth_eval = scaler.inverse_transform(synth_2d)
    else:
        synth_eval = synth_2d.copy()

    # Descobre quantas janelas (N) temos no array flat
    N = len(synth_eval) // n_steps
    
    # --- 1. Violação RMN (Ponto a ponto -> Janela) ---
    try:
        i_rt = features.index('NMRT') # Ajuste o nome se necessário (ex: NMRT_FINAL)
        i_re = features.index('NMRE')
        i_fl = features.index('NMRFL')
        
        tol = 1e-4
        # Agora a comparação é feita nas porosidades reais (ex: 0.15 <= 0.20)
        viol_re_rt = synth_eval[:, i_re] > (synth_eval[:, i_rt] + tol)
        viol_fl_re = synth_eval[:, i_fl] > (synth_eval[:, i_re] + tol)
        
        # Um ponto viola se quebrar qualquer uma das duas regras
        viol_nmr_pontos = viol_re_rt | viol_fl_re
        
        # Uma janela viola se PELO MENOS UM ponto dentro dela estiver errado
        viol_nmr_janelas = viol_nmr_pontos.reshape(N, n_steps).any(axis=1)
    except ValueError:
        print("Aviso: Curvas de RMN não encontradas.")
        viol_nmr_janelas = np.zeros(N, dtype=bool)

    # --- 2. Violação de Correlações (Janela a janela) ---
    # As correlações funcionam perfeitamente no dado desnormalizado também.
    viol_corr_janelas = np.zeros(N, dtype=bool)
    try:
        i_rhob = features.index('RHOB')
        i_nphi = features.index('NPHI')
        i_dt = features.index('DT')
        
        # Remonta o formato 3D (N, T, A) para analisar o Pearson por janela
        synth_3d = synth_eval.reshape(N, n_steps, -1)
        
        for j in range(N):
            rhob_win = synth_3d[j, :, i_rhob]
            nphi_win = synth_3d[j, :, i_nphi]
            dt_win = synth_3d[j, :, i_dt]
            
            # Correlação RHOB vs NPHI (A física exige que seja <= -0.7)
            r_rhob_nphi = np.corrcoef(rhob_win, nphi_win)[0, 1]
            if np.isnan(r_rhob_nphi) or r_rhob_nphi > -0.7:
                viol_corr_janelas[j] = True
                
            # Correlação NPHI vs DT (A física exige que seja >= 0.7)
            r_nphi_dt = np.corrcoef(nphi_win, dt_win)[0, 1]
            if np.isnan(r_nphi_dt) or r_nphi_dt < 0.7:
                viol_corr_janelas[j] = True
                
    except ValueError:
        print("Aviso: Curvas convencionais (RHOB, NPHI, DT) não encontradas.")

    
    # --- 3. Cálculo Final do PVR ---
    # União: Se a janela violou o RMN *OU* a Correlação, ela é descartada (True)
    violation_mask = viol_nmr_janelas | viol_corr_janelas
    
    pvr_score = np.mean(violation_mask) * 100.0
    
    return pvr_score, violation_mask

# ====================================================
# Função de Treino e Geração
# ====================================================
def train_and_generate(MODE, X_train, X_val, T, A, Z, BATCH_SIZE, num_examples_to_generate, features2, scaler):
    """
    Retorna: syn_2d (Original Scale), extras
    """
    extras = {}

    if MODE == "GAN":
        train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds   = tf.data.Dataset.from_tensor_slices(X_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        gan_generator = gan_lib.make_generator_1d_constrained(T, A, Z)
        discriminator = gan_lib.make_discriminator_1d_robust(T, A)

        trainer = ChemoGANTrainer(gan_generator, discriminator, features, Z_dim=Z)
        history = trainer.fit(train_ds, val_ds, epochs=3000) 

        extras["train_history"] = history
        
        # ---> AQUI: Chama a função de plotar o treinamento <---
        # Certifique-se de importar o arquivo plots.py no topo (import plots)
        plots.plot_gan_convergence(history, save_path="chemogan_convergence.png")

        seed = tf.random.normal([num_examples_to_generate, Z])
        pred_btA = gan_generator(seed, training=False).numpy()

    elif MODE == "TRANSFORMER":

        decoder_model = transformer_trainer.train_transformer(
            X_train, T, A, Z_dim=Z, epochs=3000, batch_size=BATCH_SIZE
        )

        seed = tf.random.normal([num_examples_to_generate, Z])
        pred_btA = decoder_model(seed, training=False).numpy()

    elif MODE == "DIFFUSION":

        diff_manager = diffusion_trainer.train_diffusion(
            X_train, T, A, epochs=3000, batch_size=BATCH_SIZE
        )

        pred_btA = diff_manager.generate(num_examples_to_generate, T, A)
        pred_btA = np.clip(pred_btA, 0.0, 1.0) # Garantir limites

    else:

        raise ValueError("MODE inválido")

    # Voltar para escala original para calcular métricas de geologia
    syn_2d = scaler.inverse_transform(pred_btA.reshape(-1, A))

    return syn_2d, extras

# ====================================================
# Função Principal de Execução
# ====================================================
def run_all_models(dicionario_pocos, pocos_concatenados, pocos_cegos, X, T, A, Z, BATCH_SIZE, features, scaler, n_steps):
    
    
    # 1. Dados Reais na escala original (para métricas estatísticas)
    real_2d = pocos_concatenados[features].values

    # Split
    X_train, X_val = train_test_split(X, test_size=0.15, shuffle=False, random_state=42)
    
    num_examples_to_generate = round(pocos_concatenados.shape[0] / n_steps)

    summary_rows = []

    details = {} 

    for MODE in ["GAN", "TRANSFORMER", "DIFFUSION"]:

        print(f"\n==============================")
        print(f"Rodando MODE = {MODE}")
        print(f"==============================")

        # syn_2d vem na escala ORIGINAL

        syn_2d, extras = train_and_generate(
            MODE, X_train, X_val, T, A, Z, BATCH_SIZE,
            num_examples_to_generate, features, scaler
        )     

        # (A) Distâncias por feature (Original Scale)

        df_feat = dist_metrics_per_feature(real_2d, syn_2d, features)
        details[MODE] = df_feat


        w_mean = float(df_feat["wasserstein"].mean())
        ks_mean = float(df_feat["ks_stat"].mean())

        # (B) Correlação (Original Scale)
        corr_mae = corr_error(real_2d, syn_2d)
        
        # (C) Taxa de Violação Física (PVR) e Máscara de Rejeição
        pvr_score, mask_janelas = calculate_pvr(syn_2d, features, n_steps=n_steps)

        # Expande a máscara de janelas para o formato de pontos (repetindo para os n_steps)
        mask_pontos = np.repeat(mask_janelas, n_steps)
        
        # Filtro de Rejeição Baseado em Física (Mantém apenas onde a máscara é False)
        syn_2d_filtrado = syn_2d[~mask_pontos]

        print(f"   -> [PVR] {pvr_score:.2f}% das janelas violaram a física.")
        print(f"   -> [Filtro] Sintéticos gerados: {len(syn_2d)} pts | Válidos retidos: {len(syn_2d_filtrado)} pts")

        # (D) XGBoost (Treina APENAS com os dados fisicamente válidos)
        if len(syn_2d_filtrado) > 0:
            xgb_metrics = xgb.regressao_xbg(syn_2d_filtrado, real_2d, pocos_cegos, features)
        else:
            print("   -> [Aviso] Nenhuma amostra válida sobrou para treinar o XGBoost!")
            xgb_metrics = None

        if isinstance(xgb_metrics, dict):
            rmse = xgb_metrics.get("rmse", np.nan)
            mae  = xgb_metrics.get("mae", np.nan)
            r2   = xgb_metrics.get("r2", np.nan)
            resultados = xgb_metrics.get("resultados", np.nan)
            resultados.to_excel(f"comparacao_pocos_cegos_{MODE}.xlsx", index=False)
        else:
            rmse = mae = r2 = np.nan

        summary_rows.append({
            "model": MODE,
            "wasserstein_mean": w_mean,
            "ks_mean": ks_mean,
            "corr_mae": corr_mae,
            "pvr_percent": pvr_score,
            "xgb_rmse": rmse,
            "xgb_mae": mae,
            "xgb_r2": r2
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("wasserstein_mean", ascending=True)
    return df_summary, details

# ---------------------------------------------------
# CHAMADA FINAL
# ---------------------------------------------------
df_summary, details = run_all_models(
    dicionario_pocos=dicionario_pocos,
    pocos_concatenados=pocos_concatenados,
    pocos_cegos=pocos_cegos,
    X=X, T=T, A=A, Z=Z, BATCH_SIZE=BATCH_SIZE,
    features=features, scaler=scaler, n_steps=n_steps
)
print("\n===== RESUMO FINAL (comparação dos 3 modelos) =====")
print(df_summary)