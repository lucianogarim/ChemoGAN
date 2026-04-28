# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import xgboost as xgb

# def metricas(y_true, y_pred, prefix=""):
#     y_true = np.ravel(y_true); y_pred = np.ravel(y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     r2  = r2_score(y_true, y_pred)
#     print(f'{prefix}MAE : {mae :.4f}\n{prefix}MSE : {mse :.5f}\n{prefix}R2  : {r2  :.4f}\n')
#     return {"MAE": mae, "MSE": mse, "R2": r2}

# def make_xgb():
#     return xgb.XGBRegressor(
#         booster='gbtree',
#         max_depth=8,
#         min_child_weight=3,
#         gamma=0.002159770330420502,
#         subsample=0.8368159879829131,
#         colsample_bytree=0.9600160733401647,
#         alpha=0.41285244400844556,
#         learning_rate=0.04499101368613361,
#         eval_metric='rmse',
#         early_stopping_rounds=10,
#         n_estimators=300,
#         random_state=42
#     )

# def scatter_real_vs_pred(y_train, y_pred_train, y_val, y_pred_val, title):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(np.ravel(y_train), np.ravel(y_pred_train), alpha=0.5, label='Train')
#     plt.scatter(np.ravel(y_val),   np.ravel(y_pred_val),   alpha=0.5, label='Validation')
#     mn = min(np.min(y_train), np.min(y_val), np.min(y_pred_train), np.min(y_pred_val))
#     mx = max(np.max(y_train), np.max(y_val), np.max(y_pred_train), np.max(y_pred_val))
#     plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
#     plt.xlabel('Real values'); plt.ylabel('Predictions'); plt.title(title)
#     plt.legend(); plt.grid(True); plt.show()
    

# def _get_depth_col(poco, preferred=("MD", "DEPTH", "Depth", "depth")):
#     """
#     Tenta achar automaticamente a coluna de profundidade no DataFrame do poço.
#     Você pode ajustar preferred se quiser.
#     """
#     cols = list(poco.columns)
#     for c in preferred:
#         if c in cols:
#             return c
#     return None

# def eval_wells(model, scaler2, pocos_cegos, logs, title_suffix=""):
#     """
#     Avalia poços cegos com métricas + scatter + perfil em profundidade (real vs previsto).
#     Requer que cada 'poco' tenha uma coluna de profundidade (MD ou DEPTH).
#     """
#     print(f'Métricas aplicadas aos poços de teste {title_suffix}')
#     results = {}

#     target_name = logs[-1]  # por convenção: última coluna = alvo (TARGET)

#     for nome_poco, poco in pocos_cegos.items():

#         # --- 1) descobrir coluna de profundidade ---
#         depth_col = _get_depth_col(poco)
#         if depth_col is None:
#             print(f"⚠️ {nome_poco}: não achei coluna de profundidade (MD/DEPTH). Vou fazer só scatter/métricas.")
#         else:
#             depth = poco[depth_col].to_numpy().ravel()

#         # --- 2) preparar X/y (mesma ordem de colunas) ---
#         df = poco[logs].copy()  # garante ordem
#         X_poco = df.iloc[:, :-1].to_numpy()
#         y_poco = df.iloc[:,  -1].to_numpy().ravel()

#         # se tiver NaN nas features, o scaler/modelo podem quebrar
#         # aqui a estratégia mais segura é filtrar linhas válidas
#         mask = np.isfinite(X_poco).all(axis=1) & np.isfinite(y_poco)
#         if depth_col is not None:
#             mask = mask & np.isfinite(depth)

#         if mask.sum() < 5:
#             print(f"⚠️ {nome_poco}: poucas amostras válidas após filtro (n={mask.sum()}). Pulando.")
#             continue

#         X_ok = X_poco[mask]
#         y_ok = y_poco[mask]
#         if depth_col is not None:
#             d_ok = depth[mask]

#         # --- 3) escala e predição ---
#         X_ok_scaled = scaler2.transform(X_ok)
#         y_hat = model.predict(X_ok_scaled).ravel()

#         print(f'>> {nome_poco}')
#         results[nome_poco] = metricas(y_ok, y_hat, prefix="  ")

#         # --- 4) Scatter Real vs Pred ---
#         plt.figure(figsize=(8, 6))
#         plt.scatter(y_ok, y_hat, alpha=0.5, label='Real vs Pred')
#         mn, mx = float(min(y_ok.min(), y_hat.min())), float(max(y_ok.max(), y_hat.max()))
#         plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
#         plt.xlabel('Real values'); plt.ylabel('Predictions')
#         plt.title(f'Real vs Pred — {nome_poco} {title_suffix}')
#         plt.legend(loc='upper left'); plt.grid(True); plt.show()

#         # --- 5) Perfil em profundidade (REAL vs PREVISTO) ---
#         if depth_col is not None:
#             # ordena por profundidade (garante linha contínua)
#             order = np.argsort(d_ok)
#             d2 = d_ok[order]
#             y2 = y_ok[order]
#             p2 = y_hat[order]

#             plt.figure(figsize=(4, 15))
#             plt.plot(y2, d2, label=f"{target_name} real")
#             plt.plot(p2, d2, linestyle="--", label=f"{target_name} predicted")

#             plt.gca().invert_yaxis()
#             plt.xlabel(target_name)
#             plt.ylabel(depth_col)
#             plt.title(f"{nome_poco} {title_suffix}")
#             plt.legend()
#             plt.grid(True)
#             plt.show()

#     return results



# def run_experiment(label, X, y, pocos_cegos, logs, plot_title_suffix=""):
#     """
#     Executa split, scaling, treino, validação e (opcional) avaliação em poços.
#     Retorna dicionário com métricas e objetos (modelo e scaler).
#     """
#     print(f"\n===== {label} =====")
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

#     scaler2 = MinMaxScaler()
#     X_train_scaled = scaler2.fit_transform(X_train)
#     X_val_scaled   = scaler2.transform(X_val)

#     model = make_xgb()
#     model.fit(X_train_scaled, y_train.ravel(),
#               eval_set=[(X_val_scaled, y_val.ravel())],
#               verbose=False)

#     y_pred_train = model.predict(X_train_scaled)
#     y_pred_val   = model.predict(X_val_scaled)

#     print('Métricas de treinamento')
#     m_train = metricas(y_train, y_pred_train)
#     print('Métricas de validação')
#     m_val   = metricas(y_val,   y_pred_val)

#     scatter_real_vs_pred(y_train, y_pred_train, y_val, y_pred_val,
#                          title=f'Real vs Pred — {label} {plot_title_suffix}')

#     wells_metrics = None
#     if (pocos_cegos is not None) and (logs is not None):
#         wells_metrics = eval_wells(model, scaler2, pocos_cegos, logs,
#                                    title_suffix=f'({label})')

#     return {
#         "label": label,
#         "train_metrics": m_train,
#         "val_metrics": m_val,
#         "wells_metrics": wells_metrics,
#         "model": model,
#         "scaler": scaler2
#     }

# # --- TABELA E GRÁFICOS PARA POÇOS CEGOS ---
# def resumo_wells_table(res_dicts):
#     """
#     res_dicts: lista com [res_real, res_syn, res_both] (saídas de run_experiment)
#     Retorna DataFrame com colunas: Poço, Cenário, MAE, MSE, R²
#     """
#     rows = []
#     for res in res_dicts:
#         label = res["label"]
#         wells = res["wells_metrics"]
#         if wells is None:  # caso você rode sem pocos_cegos/logs
#             continue
#         for nome_poco, metrics in wells.items():
#             rows.append({
#                 "Poço": nome_poco,
#                 "Cenário": label,
#                 "MAE": metrics["MAE"],
#                 "MSE": metrics["MSE"],
#                 "R²":  metrics["R2"],
#             })
#     return pd.DataFrame(rows)

# def plot_r2_barras(df_wells):
#     """Gráfico de barras de R² por poço e cenário."""
#     cenarios = list(df_wells["Cenário"].unique())
#     pocos    = list(df_wells["Poço"].unique())
#     x = np.arange(len(pocos))
#     width = 0.8 / max(1, len(cenarios))  # largura adaptativa

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i, cenario in enumerate(cenarios):
#         r2_vals = df_wells[df_wells["Cenário"] == cenario].set_index("Poço").loc[pocos]["R²"]
#         ax.bar(x + i*width, r2_vals, width, label=cenario)

#     ax.set_xticks(x + (len(cenarios)-1)*width/2)
#     ax.set_xticklabels(pocos, rotation=45, ha="right")
#     ax.set_ylabel("R²")
#     ax.set_title("Comparison of R² by well and scenario")
#     ax.legend()
#     ax.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.tight_layout()
#     plt.show()

# def plot_metric_barras(df_wells, metric="MAE", titulo=None):
#     """Versão genérica para MAE/MSE também."""
#     cenarios = list(df_wells["Cenário"].unique())
#     pocos    = list(df_wells["Poço"].unique())
#     x = np.arange(len(pocos))
#     width = 0.8 / max(1, len(cenarios))

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i, cenario in enumerate(cenarios):
#         vals = df_wells[df_wells["Cenário"] == cenario].set_index("Poço").loc[pocos][metric]
#         ax.bar(x + i*width, vals, width, label=cenario)

#     ax.set_xticks(x + (len(cenarios)-1)*width/2)
#     ax.set_xticklabels(pocos, rotation=45, ha="right")
#     ax.set_ylabel(metric)
#     ax.set_title(titulo or f"Comparison of {metric} by well and scenario")
#     ax.legend()
#     ax.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.tight_layout()
#     plt.show()
    

# # Resumo comparativo
# def resumo(res):
#     return (res["train_metrics"]["MAE"], res["train_metrics"]["R2"],
#             res["val_metrics"]["MAE"],   res["val_metrics"]["R2"])


# def regressao_xbg(predictions, pocos_concatenados, pocos_cegos, logs):

#     # X, y de cada cenário (última coluna = alvo)
#     X_real,   y_real   = pocos_concatenados[:, :-1], pocos_concatenados[:,  -1]
#     X_synth,  y_synth  = predictions[:,        :-1], predictions[:,         -1]
#     X_both    = np.concatenate([X_synth, X_real], axis=0)
#     y_both    = np.concatenate([y_synth, y_real], axis=0)
    
#     # =========================
#     # 2) EXECUÇÃO DOS CENÁRIOS
#     # =========================
#     res_real  = run_experiment("Real",        X_real,  y_real,  pocos_cegos, logs)
#     res_both  = run_experiment("Real + Synthetic", X_both, y_both, pocos_cegos, logs)
    
#     # --- Construir a tabela e visualizar ---
#     df_wells = resumo_wells_table([res_real, res_both])

#     # ver a tabela “longa”
#     print(df_wells.sort_values(["Poço", "Cenário"]).to_string(index=False))

#     # (opcional) pivot para comparar lado a lado
#     df_pivot = df_wells.pivot_table(index="Poço", columns="Cenário", values=["MAE", "MSE", "R²"])
#     print("\n== Pivot by well (MAE/MSE/R² by scenario) ==")
#     print(df_pivot)

#     # (opcional) salvar
#     df_wells.to_csv("comparacao_pocos_cegos.csv", index=False)

#     # --- Gráficos ---
#     plot_r2_barras(df_wells)  # R² por poço e cenário

#     # Se quiser também MAE e MSE:
#     plot_metric_barras(df_wells, metric="MAE", titulo="MAE by well and scenario")
#     plot_metric_barras(df_wells, metric="MSE", titulo="MSE by well and scenario")
    
#     A = resumo(res_real);  B = resumo(res_both)
#     print("\n=== RESUMO (MAE, R2) — train | validation ===")
#     print(f"A) Real         : MAE_tr={A[0]:.4f}, R2_tr={A[1]:.3f} | MAE_val={A[2]:.4f}, R2_val={A[3]:.3f}")
#     print(f"B) Real + Synthetic    : MAE_tr={B[0]:.4f}, R2_tr={B[1]:.3f} | MAE_val={B[2]:.4f}, R2_val={B[3]:.3f}")

    
#     return 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import uniform, randint
import warnings

# Suprimir warnings chatos do XGBoost sobre early_stopping
warnings.filterwarnings("ignore", category=UserWarning)

def metricas(y_true, y_pred, prefix=""):
    y_true = np.ravel(y_true); y_pred = np.ravel(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f'{prefix}MAE : {mae :.4f}\n{prefix}MSE : {mse :.5f}\n{prefix}R2  : {r2  :.4f}\n')
    return {"MAE": mae, "MSE": mse, "R2": r2}

def optimize_xgb(X_train, y_train):
    """
    Realiza uma busca randômica pelos melhores hiperparâmetros do XGBoost.
    """
    print("   -> Iniciando busca de hiperparâmetros (RandomizedSearchCV)...")
    base_model = xgb.XGBRegressor(booster='gbtree', random_state=42, eval_metric='rmse')

    # Espaço de busca focado em evitar overfitting em dados geológicos complexos
    param_dist = {
        'max_depth': randint(3, 20),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0.0, 0.5),
        'subsample': uniform(0.6, 0.4),        # Varia de 0.6 a 1.0
        'colsample_bytree': uniform(0.6, 0.4), # Varia de 0.6 a 1.0
        'learning_rate': uniform(0.01, 0.1),
        'n_estimators': randint(10, 400)
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter = 150, # Número de combinações testadas (aumente se quiser refinar mais)
        scoring='neg_mean_squared_error',
        cv=3,      # Validação cruzada com 3 folds
        verbose=0,
        n_jobs=-1, # Usa todos os núcleos da CPU
        random_state=42
    )

    search.fit(X_train, y_train)
    print(f"   -> Melhores parâmetros encontrados: {search.best_params_}")
    return search.best_params_

def make_xgb(best_params=None):
    """
    Cria o modelo com os parâmetros otimizados ou usa os padrões originais como fallback.
    """
    if best_params:
        return xgb.XGBRegressor(
            booster='gbtree',
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=10, # <--- ADICIONADO AQUI
            **best_params
        )
    else:
        # Fallback de segurança
        return xgb.XGBRegressor(
            booster='gbtree',
            max_depth=10,
            min_child_weight=3,
            gamma=0.002159770330420502,
            subsample=0.8368159879829131,
            colsample_bytree=0.9600160733401647,
            alpha=0.41285244400844556,
            learning_rate=0.04499101368613361,
            eval_metric='rmse',
            early_stopping_rounds=10, # Já estava aqui
            n_estimators=300,
            random_state=42
        )

def scatter_real_vs_pred(y_train, y_pred_train, y_val, y_pred_val, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.ravel(y_train), np.ravel(y_pred_train), alpha=0.5, label='Train')
    plt.scatter(np.ravel(y_val),   np.ravel(y_pred_val),   alpha=0.5, label='Validation')
    mn = min(np.min(y_train), np.min(y_val), np.min(y_pred_train), np.min(y_pred_val))
    mx = max(np.max(y_train), np.max(y_val), np.max(y_pred_train), np.max(y_pred_val))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel('Real values'); plt.ylabel('Predictions'); plt.title(title)
    plt.legend(); plt.grid(True); plt.show()
    

def _get_depth_col(poco, preferred=("MD", "DEPTH", "Depth", "depth")):
    """
    Tenta achar automaticamente a coluna de profundidade no DataFrame do poço.
    """
    cols = list(poco.columns)
    for c in preferred:
        if c in cols:
            return c
    return None

def eval_wells(model, scaler2, pocos_cegos, logs, title_suffix=""):
    """
    Avalia poços cegos com métricas + scatter + perfil em profundidade (real vs previsto).
    """
    print(f'Métricas aplicadas aos poços de teste {title_suffix}')
    results = {}
    target_name = logs[-1]  # por convenção: última coluna = alvo (TARGET)

    for nome_poco, poco in pocos_cegos.items():
        # --- 1) descobrir coluna de profundidade ---
        depth_col = _get_depth_col(poco)
        if depth_col is None:
            print(f"⚠️ {nome_poco}: não achei coluna de profundidade (MD/DEPTH). Vou fazer só scatter/métricas.")
        else:
            depth = poco[depth_col].to_numpy().ravel()

        # --- 2) preparar X/y (mesma ordem de colunas) ---
        df = poco[logs].copy()  
        X_poco = df.iloc[:, :-1].to_numpy()
        y_poco = df.iloc[:,  -1].to_numpy().ravel()

        mask = np.isfinite(X_poco).all(axis=1) & np.isfinite(y_poco)
        if depth_col is not None:
            mask = mask & np.isfinite(depth)

        if mask.sum() < 5:
            print(f"⚠️ {nome_poco}: poucas amostras válidas após filtro (n={mask.sum()}). Pulando.")
            continue

        X_ok = X_poco[mask]
        y_ok = y_poco[mask]
        if depth_col is not None:
            d_ok = depth[mask]

        # --- 3) escala e predição ---
        X_ok_scaled = scaler2.transform(X_ok)
        y_hat = model.predict(X_ok_scaled).ravel()

        print(f'>> {nome_poco}')
        results[nome_poco] = metricas(y_ok, y_hat, prefix="  ")

        # --- 4) Scatter Real vs Pred ---
        plt.figure(figsize=(8, 6))
        plt.scatter(y_ok, y_hat, alpha=0.5, label='Real vs Pred')
        mn, mx = float(min(y_ok.min(), y_hat.min())), float(max(y_ok.max(), y_hat.max()))
        plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
        plt.xlabel('Real values'); plt.ylabel('Predictions')
        plt.title(f'Real vs Pred — {nome_poco} {title_suffix}')
        plt.legend(loc='upper left'); plt.grid(True); plt.show()

        # --- 5) Perfil em profundidade (REAL vs PREVISTO) ---
        if depth_col is not None:
            order = np.argsort(d_ok)
            d2 = d_ok[order]
            y2 = y_ok[order]
            p2 = y_hat[order]

            plt.figure(figsize=(4, 15))
            plt.plot(y2, d2, label=f"{target_name} real")
            plt.plot(p2, d2, linestyle="--", label=f"{target_name} predicted")

            plt.gca().invert_yaxis()
            plt.xlabel(target_name)
            plt.ylabel(depth_col)
            plt.title(f"{nome_poco} {title_suffix}")
            plt.legend()
            plt.grid(True)
            plt.show()

    return results

def run_experiment(label, X, y, pocos_cegos, logs, plot_title_suffix="", optimize=True):
    """
    Executa split, scaling, treino, validação e avaliação em poços cegos.
    Otimiza hiperparâmetros automaticamente se optimize=True.
    """
    print(f"\n===== {label} =====")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

    scaler2 = MinMaxScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_val_scaled   = scaler2.transform(X_val)

    if optimize:
        best_params = optimize_xgb(X_train_scaled, y_train.ravel())
        model = make_xgb(best_params)
        
        # EARLY STOPPING REMOVIDO DO FIT
        model.fit(X_train_scaled, y_train.ravel(),
                  eval_set=[(X_val_scaled, y_val.ravel())],
                  verbose=False)
    else:
        model = make_xgb()
        model.fit(X_train_scaled, y_train.ravel(),
                  eval_set=[(X_val_scaled, y_val.ravel())],
                  verbose=False)


    y_pred_train = model.predict(X_train_scaled)
    y_pred_val   = model.predict(X_val_scaled)

    print('Métricas de treinamento')
    m_train = metricas(y_train, y_pred_train)
    print('Métricas de validação')
    m_val   = metricas(y_val,   y_pred_val)

    scatter_real_vs_pred(y_train, y_pred_train, y_val, y_pred_val,
                         title=f'Real vs Pred — {label} {plot_title_suffix}')

    wells_metrics = None
    if (pocos_cegos is not None) and (logs is not None):
        wells_metrics = eval_wells(model, scaler2, pocos_cegos, logs,
                                   title_suffix=f'({label})')

    return {
        "label": label,
        "train_metrics": m_train,
        "val_metrics": m_val,
        "wells_metrics": wells_metrics,
        "model": model,
        "scaler": scaler2
    }

# --- TABELA E GRÁFICOS PARA POÇOS CEGOS ---
def resumo_wells_table(res_dicts):
    rows = []
    for res in res_dicts:
        label = res["label"]
        wells = res["wells_metrics"]
        if wells is None: 
            continue
        for nome_poco, metrics in wells.items():
            rows.append({
                "Poço": nome_poco,
                "Cenário": label,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "R²":  metrics["R2"],
            })
    return pd.DataFrame(rows)

def plot_r2_barras(df_wells):
    cenarios = list(df_wells["Cenário"].unique())
    pocos    = list(df_wells["Poço"].unique())
    x = np.arange(len(pocos))
    width = 0.8 / max(1, len(cenarios)) 

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cenario in enumerate(cenarios):
        r2_vals = df_wells[df_wells["Cenário"] == cenario].set_index("Poço").loc[pocos]["R²"]
        ax.bar(x + i*width, r2_vals, width, label=cenario)

    ax.set_xticks(x + (len(cenarios)-1)*width/2)
    ax.set_xticklabels(pocos, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Comparison of R² by well and scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_metric_barras(df_wells, metric="MAE", titulo=None):
    cenarios = list(df_wells["Cenário"].unique())
    pocos    = list(df_wells["Poço"].unique())
    x = np.arange(len(pocos))
    width = 0.8 / max(1, len(cenarios))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cenario in enumerate(cenarios):
        vals = df_wells[df_wells["Cenário"] == cenario].set_index("Poço").loc[pocos][metric]
        ax.bar(x + i*width, vals, width, label=cenario)

    ax.set_xticks(x + (len(cenarios)-1)*width/2)
    ax.set_xticklabels(pocos, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(titulo or f"Comparison of {metric} by well and scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def resumo(res):
    return (res["train_metrics"]["MAE"], res["train_metrics"]["R2"],
            res["val_metrics"]["MAE"],   res["val_metrics"]["R2"])


def regressao_xbg(predictions_filtradas, pocos_concatenados, pocos_cegos, logs):
    """
    Função principal. predictions_filtradas já devem vir limpas de alucinações (Rejection Sampling).
    """
    # X, y de cada cenário (última coluna = alvo)
    X_real,   y_real   = pocos_concatenados[:, :-1], pocos_concatenados[:,  -1]
    X_synth,  y_synth  = predictions_filtradas[:, :-1], predictions_filtradas[:, -1]
    X_both    = np.concatenate([X_synth, X_real], axis=0)
    y_both    = np.concatenate([y_synth, y_real], axis=0)
    
    # =========================
    # 2) EXECUÇÃO DOS CENÁRIOS
    # =========================
    # Otimiza o XGBoost tanto para os dados originais quanto para os combinados
    res_real  = run_experiment("Real", X_real, y_real, pocos_cegos, logs, optimize=True)
    res_both  = run_experiment("Real + Synthetic", X_both, y_both, pocos_cegos, logs, optimize=True)
    
    # --- Construir a tabela e visualizar ---
    df_wells = resumo_wells_table([res_real, res_both])

    print("\n" + df_wells.sort_values(["Poço", "Cenário"]).to_string(index=False))

    df_pivot = df_wells.pivot_table(index="Poço", columns="Cenário", values=["MAE", "MSE", "R²"])
    print("\n== Pivot by well (MAE/MSE/R² by scenario) ==")
    print(df_pivot)

    df_wells.to_csv("comparacao_pocos_cegos.csv", index=False)

    plot_r2_barras(df_wells) 
    plot_metric_barras(df_wells, metric="MAE", titulo="MAE by well and scenario")
    plot_metric_barras(df_wells, metric="MSE", titulo="MSE by well and scenario")
    
    A = resumo(res_real);  B = resumo(res_both)
    print("\n=== RESUMO (MAE, R2) — train | validation ===")
    print(f"A) Real          : MAE_tr={A[0]:.4f}, R2_tr={A[1]:.3f} | MAE_val={A[2]:.4f}, R2_val={A[3]:.3f}")
    print(f"B) Real + Synth  : MAE_tr={B[0]:.4f}, R2_tr={B[1]:.3f} | MAE_val={B[2]:.4f}, R2_val={B[3]:.3f}")
    
    # Retorna o RMSE e R2 do cenário "Real + Synthetic" no poço de validação global 
    # para você plotar no summary geral (df_summary) se quiser
    return {
        "mae": B[2],
        "rmse": np.sqrt(B[2]), # Aproximação se quiser retornar algo
        "r2": B[3],
        "resultados": df_wells
    }





























