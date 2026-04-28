
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import pandas as pd
import lasio
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

def carregamento(data_dir, mnemonicos):

    dicionario_pocos = {}
    
    # Lista todos os arquivos no diretório
    arquivos = os.listdir(data_dir)
    
    # Itera sobre os arquivos
    for arquivo in arquivos:
        # Verifica se o arquivo possui a extensão .las
        if arquivo.endswith(".las"):
            caminho_arquivo = os.path.join(data_dir, arquivo)
            # Lê o arquivo .las usando o módulo lasio
            las = lasio.read(caminho_arquivo)
            dicionario_pocos[arquivo] = las.df().reset_index()  # Agregando valores ao dicionário
    
    
    aux = dicionario_pocos.copy()
    for nome_poco, poco in dicionario_pocos.items():
        try:
            aux[nome_poco] = aux[nome_poco].loc[:,mnemonicos]
        except Exception as e:
            print("O poço {} não possui os perfis {}".format(nome_poco, str(e)))
            aux.pop(nome_poco);
    dicionario_pocos = aux.copy()
    
    return dicionario_pocos
    
def filtro_profundidade(dicionario_pocos, filtros):
 
    for nome_poco, (prof_min, prof_max) in filtros.items():
        depth = dicionario_pocos[nome_poco]['DEPTH']
        indices = (depth>prof_min) & (depth<prof_max)
        dicionario_pocos[nome_poco] = dicionario_pocos[nome_poco][indices]
        
    return dicionario_pocos

def trata_outliers(dicionario_pocos, logs):
    
    for nome_poco, poco in dicionario_pocos.items():
        # Tratamento para nan
        dicionario_pocos[nome_poco].dropna(inplace = True)
        model_IF = IsolationForest(contamination=0.03, random_state=42)
        model_IF.fit(dicionario_pocos[nome_poco][logs])
        dicionario_pocos[nome_poco]['anomaly'] = model_IF.predict(dicionario_pocos[nome_poco][logs])
        dicionario_pocos[nome_poco] = dicionario_pocos[nome_poco][dicionario_pocos[nome_poco]['anomaly'] != -1]
        
    return dicionario_pocos

def escolhe_pocos_cegos(dicionario_pocos, lista_pocos_cegos):
    
    # Escolha dos poços cegos
    pocos_cegos = {chave: dicionario_pocos[chave].dropna() for chave in lista_pocos_cegos}
    # Retira os poços do dataset
    for chave in pocos_cegos.keys():
       dicionario_pocos.pop(chave)

    return dicionario_pocos, pocos_cegos

def concatena_pocos(dicionario_pocos):
    
    pocos_concatenados = []
    for nome_poco, poco in dicionario_pocos.items():
        pocos_concatenados.append(poco)
    
    pocos_concatenados = pd.concat(pocos_concatenados, ignore_index=True).dropna()
    
    return pocos_concatenados
    

def muda_escala(pocos_concatenados, logs):
    
    # Padronização dos dados
    scaler = MinMaxScaler()
    pocos_concatenados = scaler.fit_transform(pocos_concatenados[logs])
    
    return pocos_concatenados, scaler
    
    

    
def cria_sequencias(dicionario_pocos, features, n_steps, scaler):
    """
    Gera sequências temporais respeitando as fronteiras de cada poço.
    
    Args:
        dicionario_pocos (dict): Dicionário contendo os DataFrames dos poços.
        features (list): Lista de mnemonicos (colunas) a serem usadas.
        n_steps (int): Tamanho da janela temporal (ex: 64).
        scaler (MinMaxScaler): Scaler JÁ FITADO nos dados globais.
        
    Returns:
        np.array: Array 3D com shape (N_amostras, n_steps, n_features)
    """
    X_lista = []
    
    print(f"Gerando sequências para {len(dicionario_pocos)} poços...")
    
    for nome_poco, df in dicionario_pocos.items():
        # 1. Selecionar apenas as colunas desejadas (features)
        dados_brutos = df[features].values
        
        # 2. Aplicar a escala usando o scaler global (transform, NÃO fit)
        # Isso garante que a normalização seja consistente com o treino global
        dados_scaled = scaler.transform(dados_brutos)
        
        # 3. Validar se o poço tem tamanho suficiente
        tamanho_poco = len(dados_scaled)
        if tamanho_poco < n_steps:
            print(f"Aviso: Poço {nome_poco} é muito curto ({tamanho_poco}) e será ignorado.")
            continue
            
        # 4. Criar as sequências APENAS dentro deste poço
        # Usamos range para criar uma lista de índices e fatiar
        for i in range(0, tamanho_poco - n_steps + 1, 10):
            seq = dados_scaled[i : i + n_steps, :]
            X_lista.append(seq)

            
    # Converter lista de arrays para um único array numpy 3D
    X_final = np.array(X_lista)
    
    print(f"Sequências geradas com sucesso. Shape final: {X_final.shape}")
    return X_final   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    