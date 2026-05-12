"""
Spyder Editor

Temporary script file.
"""

import os

import lasio
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def carregamento(data_dir, mnemonicos):
    """
    Load LAS files from a directory and keep only the requested mnemonics.

    Parameters
    ----------
    data_dir : str
        Directory containing the .las files.
    mnemonicos : list of str
        List of required log mnemonics/columns to keep in each well DataFrame.

    Returns
    -------
    dict
        Dictionary where each key is a LAS filename and each value is a
        pandas DataFrame containing the selected mnemonics.

    Notes
    -----
    The function name and parameter names are intentionally kept in Portuguese
    to preserve compatibility with the rest of the project.
    """
    dicionario_pocos = {}

    # List all files in the directory.
    arquivos = os.listdir(data_dir)

    # Iterate over files and load only LAS files.
    for arquivo in arquivos:
        if arquivo.endswith(".las"):
            caminho_arquivo = os.path.join(data_dir, arquivo)

            # Read the .las file using lasio.
            las = lasio.read(caminho_arquivo)
            dicionario_pocos[arquivo] = las.df().reset_index()

    aux = dicionario_pocos.copy()

    for nome_poco, poco in dicionario_pocos.items():
        try:
            aux[nome_poco] = aux[nome_poco].loc[:, mnemonicos]
        except Exception as e:
            print("Well {} does not contain the required logs {}".format(nome_poco, str(e)))
            aux.pop(nome_poco)

    dicionario_pocos = aux.copy()

    return dicionario_pocos


def filtro_profundidade(dicionario_pocos, filtros):
    """
    Filter each well by its configured depth interval.

    Parameters
    ----------
    dicionario_pocos : dict
        Dictionary of well DataFrames.
    filtros : dict
        Dictionary mapping well names to tuples of the form
        (minimum_depth, maximum_depth).

    Returns
    -------
    dict
        Dictionary of filtered well DataFrames.
    """
    for nome_poco, (prof_min, prof_max) in filtros.items():
        depth = dicionario_pocos[nome_poco]["DEPTH"]
        indices = (depth > prof_min) & (depth < prof_max)
        dicionario_pocos[nome_poco] = dicionario_pocos[nome_poco][indices]

    return dicionario_pocos


def trata_outliers(dicionario_pocos, logs):
    """
    Remove missing values and filter outliers in each well using Isolation Forest.

    Parameters
    ----------
    dicionario_pocos : dict
        Dictionary of well DataFrames.
    logs : list of str
        Log columns used to fit the Isolation Forest model.

    Returns
    -------
    dict
        Dictionary of cleaned well DataFrames.
    """
    for nome_poco, poco in dicionario_pocos.items():
        # NaN handling.
        dicionario_pocos[nome_poco].dropna(inplace=True)

        model_IF = IsolationForest(contamination=0.03, random_state=42)
        model_IF.fit(dicionario_pocos[nome_poco][logs])

        dicionario_pocos[nome_poco]["anomaly"] = model_IF.predict(dicionario_pocos[nome_poco][logs])
        dicionario_pocos[nome_poco] = dicionario_pocos[nome_poco][dicionario_pocos[nome_poco]["anomaly"] != -1]

    return dicionario_pocos


def escolhe_pocos_cegos(dicionario_pocos, lista_pocos_cegos):
    """
    Select blind wells and remove them from the training dictionary.

    Parameters
    ----------
    dicionario_pocos : dict
        Dictionary containing all available wells.
    lista_pocos_cegos : list of str
        Names of wells to be used as blind wells.

    Returns
    -------
    tuple
        A tuple containing:
        - the remaining dictionary after removing blind wells;
        - a dictionary containing only the selected blind wells.
    """
    # Select blind wells.
    pocos_cegos = {chave: dicionario_pocos[chave].dropna() for chave in lista_pocos_cegos}

    # Remove blind wells from the training dataset.
    for chave in pocos_cegos.keys():
        dicionario_pocos.pop(chave)

    return dicionario_pocos, pocos_cegos


def concatena_pocos(dicionario_pocos):
    """
    Concatenate all well DataFrames into a single DataFrame.

    Parameters
    ----------
    dicionario_pocos : dict
        Dictionary of well DataFrames.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with missing values removed.
    """
    pocos_concatenados = []

    for nome_poco, poco in dicionario_pocos.items():
        pocos_concatenados.append(poco)

    pocos_concatenados = pd.concat(pocos_concatenados, ignore_index=True).dropna()

    return pocos_concatenados


def muda_escala(pocos_concatenados, logs):
    """
    Apply MinMax scaling to the selected log columns.

    Parameters
    ----------
    pocos_concatenados : pandas.DataFrame
        Concatenated well-log DataFrame.
    logs : list of str
        Columns to be scaled.

    Returns
    -------
    tuple
        A tuple containing:
        - the scaled array;
        - the fitted MinMaxScaler instance.
    """
    # Data normalization.
    scaler = MinMaxScaler()
    pocos_concatenados = scaler.fit_transform(pocos_concatenados[logs])

    return pocos_concatenados, scaler


def cria_sequencias(dicionario_pocos, features, n_steps, scaler):
    """
    Generate sequential windows while respecting individual well boundaries.

    Parameters
    ----------
    dicionario_pocos : dict
        Dictionary containing well DataFrames.
    features : list of str
        List of mnemonics/columns to use as model features.
    n_steps : int
        Window size, for example 64.
    scaler : MinMaxScaler
        Globally fitted scaler. This function only calls transform, never fit.

    Returns
    -------
    numpy.ndarray
        3D array with shape (n_samples, n_steps, n_features).
    """
    X_lista = []

    print(f"Generating sequences for {len(dicionario_pocos)} wells...")

    for nome_poco, df in dicionario_pocos.items():
        # 1. Select only the desired columns/features.
        dados_brutos = df[features].values

        # 2. Apply the global scaler using transform, not fit.
        # This keeps normalization consistent with the global training data.
        dados_scaled = scaler.transform(dados_brutos)

        # 3. Check whether the well is long enough.
        tamanho_poco = len(dados_scaled)
        if tamanho_poco < n_steps:
            print(f"Warning: Well {nome_poco} is too short ({tamanho_poco}) and will be ignored.")
            continue

        # 4. Create sequences only within this well boundary.
        # The step of 10 is preserved from the original implementation.
        for i in range(0, tamanho_poco - n_steps + 1, 10):
            seq = dados_scaled[i: i + n_steps, :]
            X_lista.append(seq)

    # Convert the list of arrays into a single 3D NumPy array.
    X_final = np.array(X_lista)

    print(f"Sequences generated successfully. Final shape: {X_final.shape}")
    return X_final
