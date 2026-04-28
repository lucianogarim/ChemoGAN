# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:42:02 2025

@author: LUCIANOGARIM
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
    
def plot_curvas_completas(predictions,pocos_concatenados_orig, features):
    # Plotagem
    num_colunas = 2
    num_linhas = len(features) // num_colunas
    if len(features) % num_colunas != 0:
        num_linhas += 1
    
    fig, axes = plt.subplots(nrows=num_linhas, ncols=num_colunas, figsize=(15, 4 * num_linhas))
    
    for i, curva in enumerate(features):
        linha = i // num_colunas
        coluna = i % num_colunas
    
        # Plotar o histograma para Sintético (Red)
        axes[linha, coluna].plot(predictions[:, i], color='red', alpha=0.7, label='Sintético')
        axes[linha, coluna].set_title(f'Histogram of {curva}')
    
        # Plotar o histograma para Real (Blue)
        axes[linha, coluna].plot(pocos_concatenados_orig[:, i], color='blue', alpha=0.5, label='Real')
        axes[linha, coluna].legend()
    
    plt.tight_layout()
    plt.show()
    
    return

def histogramas(predictions, pocos_concatenados, features):
    # Número de colunas nos subplots
    num_colunas = 2
    
    # Calcular o número de linhas necessárias
    num_linhas = len(features) // num_colunas
    if len(features) % num_colunas != 0:
        num_linhas += 1
    
    # Criar subplots
    fig, axes = plt.subplots(nrows=num_linhas, ncols=num_colunas, figsize=(15, 4 * num_linhas))
    
    # Iterar sobre as curvas petrofísicas e criar subplots
    for i, curva in enumerate(features):
        linha = i // num_colunas
        coluna = i % num_colunas
    
        # Plotar o histograma para X_train
        axes[linha, coluna].hist(predictions[:, i], bins=20, alpha=0.7, color='red')
        axes[linha, coluna].set_title(f'Histogram of {curva}')
        axes[linha, coluna].set_xlabel('Values')
        axes[linha, coluna].set_ylabel('Frequency')
    
        # Plotar o histograma para logs_petrofisicos
        axes[linha, coluna].hist(pocos_concatenados[:, i], bins=20, alpha=0.5, color='blue')
        axes[linha, coluna].set_title(f'Histogram of {curva}')
        axes[linha, coluna].set_xlabel('Values')
        axes[linha, coluna].set_ylabel('Frequency')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar os subplots
    plt.show()
    
    return

def plot_pca(predictions, pocos_concatenados):
    
    n_components = 2
    pca = PCA(n_components=n_components)

    
    #The fit of the methods must be done only using the real sequential data
    pca.fit(pocos_concatenados)
    
    pca_real = pd.DataFrame(pca.transform(pocos_concatenados))
    pca_synth = pd.DataFrame(pca.transform(predictions))
    
    
    fig = plt.figure(constrained_layout=True, figsize=(20,10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    
    ax = fig.add_subplot(spec[0,0])
    ax.set_title('PCA results',
                 fontsize=20,
                 color='red',
                 pad=10)
    
    #PCA scatter plot
    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:,1].values,
                c='blue', alpha=0.2, label='Original')
    plt.scatter(pca_synth.iloc[:,0], pca_synth.iloc[:,1],
                c='red', alpha=0.2, label='Synthetic')
    ax.legend()
    
    fig.suptitle('Validating synthetic vs real data diversity and distributions',
                 fontsize=16,
                 color='grey')
    plt.show()
    
    return


def correlacoes(predictions, logs, pocos_concatenados):


    plt.figure(figsize=(16, 6))
    
    # Calcula a matriz de correlação
    corr_matrix = np.corrcoef(predictions, rowvar=False)
    
    # Cria o heatmap com rótulos personalizados
    sns.heatmap(corr_matrix, 
                vmin=-1, 
                vmax=1, 
                annot=True, 
                cmap='coolwarm',
                xticklabels=logs,  # Define os rótulos das colunas
                yticklabels=logs)  # Define os rótulos das linhas
    
    plt.title('Correlation Matrix of Synthetic Logs')
    plt.show()
    
    
    plt.figure(figsize=(16, 6))

    # Calcula a matriz de correlação
    corr_matrix = np.corrcoef(pocos_concatenados, rowvar=False)
    
    # Cria o heatmap com rótulos personalizados
    sns.heatmap(corr_matrix, 
                vmin=-1, 
                vmax=1, 
                annot=True, 
                cmap='coolwarm',
                xticklabels=logs,  # Define os rótulos das colunas
                yticklabels=logs)  # Define os rótulos das linhas
    
    plt.title('Correlation Matrix of Real Logs')
    plt.show()
    
    return


def plot_gan_convergence(history, save_path=None):
    """
    Generates a panel with the adversarial and physical convergence of ChemoGAN.
    Ideal for inclusion in the results section of the paper.
    """
    epochs = range(1, len(history["g_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel 1: Adversarial Dynamics ---
    axes[0].plot(epochs, history["d_loss"], label="Discriminator Loss", color='darkred', linewidth=2)
    axes[0].plot(epochs, history["g_adv"], label="Generator Adv Loss", color='navy', linewidth=2)
    axes[0].plot(epochs, history["val_g_loss"], label="Val Generator Tot", color='orange', linestyle='dashed', alpha=0.8)
    axes[0].set_title("(a) Adversarial Dynamics and Validation", fontsize=14)
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("Loss (Logistic)", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2: Physical Convergence (Physics-Informed) ---
    axes[1].plot(epochs, history["nmr_loss"], label=r"$\mathcal{L}_{NMR}$ (Porosity Chain)", color='forestgreen', linewidth=2)
    axes[1].plot(epochs, history["corr_rhob_nphi"], label=r"$\mathcal{L}_{Corr}$ ($\rho_b$ vs $\phi_N$)", color='purple', linewidth=2)
    axes[1].plot(epochs, history["corr_dt_nphi"], label=r"$\mathcal{L}_{Corr}$ ($\Delta t$ vs $\phi_N$)", color='teal', linewidth=2)
    
    # --- NOVO: Adicionando a plotagem da penalidade geoquímica ---
    if "pe_ca_loss" in history:
        axes[1].plot(epochs, history["pe_ca_loss"], label=r"$\mathcal{L}_{Corr}$ (PE vs DWSI)", color='crimson', linewidth=2)
    
    axes[1].set_title("(b) Convergence of Physical Constraints", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("Penalty Loss", fontsize=12)
    axes[1].set_yscale('log') # Log scale helps a lot to see the constraints approaching zero
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {save_path}")
    
    plt.show()





























