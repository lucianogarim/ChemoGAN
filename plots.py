# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:42:02 2025

@author: LUCIANOGARIM
"""

import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plot_curvas_completas(predictions, pocos_concatenados_orig, features):
    """
    Plot complete real and synthetic curves for each selected feature.

    Parameters
    ----------
    predictions : numpy.ndarray
        Synthetic data array with shape (n_samples, n_features).
    pocos_concatenados_orig : numpy.ndarray
        Real concatenated data array with shape (n_samples, n_features).
    features : list of str
        Feature names used to label each subplot.

    Notes
    -----
    The function name and some parameter names are kept unchanged to preserve
    compatibility with the rest of the project.
    """
    # Plot configuration.
    num_colunas = 2
    num_linhas = len(features) // num_colunas

    if len(features) % num_colunas != 0:
        num_linhas += 1

    fig, axes = plt.subplots(
        nrows=num_linhas,
        ncols=num_colunas,
        figsize=(15, 4 * num_linhas),
    )

    for i, curva in enumerate(features):
        linha = i // num_colunas
        coluna = i % num_colunas

        # Plot synthetic curve in red.
        axes[linha, coluna].plot(predictions[:, i], color="red", alpha=0.7, label="Synthetic")
        axes[linha, coluna].set_title(f"Histogram of {curva}")

        # Plot real curve in blue.
        axes[linha, coluna].plot(pocos_concatenados_orig[:, i], color="blue", alpha=0.5, label="Real")
        axes[linha, coluna].legend()

    plt.tight_layout()
    plt.show()

    return


def histogramas(predictions, pocos_concatenados, features):
    """
    Plot histograms comparing synthetic and real data for each feature.

    Parameters
    ----------
    predictions : numpy.ndarray
        Synthetic data array with shape (n_samples, n_features).
    pocos_concatenados : numpy.ndarray
        Real concatenated data array with shape (n_samples, n_features).
    features : list of str
        Feature names used to label each subplot.
    """
    # Number of columns in the subplot grid.
    num_colunas = 2

    # Calculate the required number of rows.
    num_linhas = len(features) // num_colunas
    if len(features) % num_colunas != 0:
        num_linhas += 1

    # Create subplots.
    fig, axes = plt.subplots(
        nrows=num_linhas,
        ncols=num_colunas,
        figsize=(15, 4 * num_linhas),
    )

    # Iterate over petrophysical curves and create one subplot per feature.
    for i, curva in enumerate(features):
        linha = i // num_colunas
        coluna = i % num_colunas

        # Plot histogram for synthetic data.
        axes[linha, coluna].hist(predictions[:, i], bins=20, alpha=0.7, color="red")
        axes[linha, coluna].set_title(f"Histogram of {curva}")
        axes[linha, coluna].set_xlabel("Values")
        axes[linha, coluna].set_ylabel("Frequency")

        # Plot histogram for real data.
        axes[linha, coluna].hist(pocos_concatenados[:, i], bins=20, alpha=0.5, color="blue")
        axes[linha, coluna].set_title(f"Histogram of {curva}")
        axes[linha, coluna].set_xlabel("Values")
        axes[linha, coluna].set_ylabel("Frequency")

    # Adjust layout.
    plt.tight_layout()

    # Show subplots.
    plt.show()

    return


def plot_pca(predictions, pocos_concatenados):
    """
    Compare real and synthetic data in a two-dimensional PCA projection.

    Parameters
    ----------
    predictions : numpy.ndarray
        Synthetic data array.
    pocos_concatenados : numpy.ndarray
        Real concatenated data array used to fit PCA.

    Notes
    -----
    PCA is fitted only on the real data and then applied to both real and
    synthetic data. This preserves the original validation logic.
    """
    n_components = 2
    pca = PCA(n_components=n_components)

    # The fit must be performed only using the real sequential data.
    pca.fit(pocos_concatenados)

    pca_real = pd.DataFrame(pca.transform(pocos_concatenados))
    pca_synth = pd.DataFrame(pca.transform(predictions))

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[0, 0])
    ax.set_title(
        "PCA results",
        fontsize=20,
        color="red",
        pad=10,
    )

    # PCA scatter plot.
    plt.scatter(
        pca_real.iloc[:, 0].values,
        pca_real.iloc[:, 1].values,
        c="blue",
        alpha=0.2,
        label="Original",
    )
    plt.scatter(
        pca_synth.iloc[:, 0],
        pca_synth.iloc[:, 1],
        c="red",
        alpha=0.2,
        label="Synthetic",
    )
    ax.legend()

    fig.suptitle(
        "Validating synthetic vs real data diversity and distributions",
        fontsize=16,
        color="grey",
    )
    plt.show()

    return


def correlacoes(predictions, logs, pocos_concatenados):
    """
    Plot correlation matrices for synthetic and real logs.

    Parameters
    ----------
    predictions : numpy.ndarray
        Synthetic data array.
    logs : list of str
        Log names used as heatmap labels.
    pocos_concatenados : numpy.ndarray
        Real concatenated data array.
    """
    plt.figure(figsize=(16, 6))

    # Compute the synthetic correlation matrix.
    corr_matrix = np.corrcoef(predictions, rowvar=False)

    # Create the heatmap with custom labels.
    sns.heatmap(
        corr_matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="coolwarm",
        xticklabels=logs,
        yticklabels=logs,
    )

    plt.title("Correlation Matrix of Synthetic Logs")
    plt.show()

    plt.figure(figsize=(16, 6))

    # Compute the real correlation matrix.
    corr_matrix = np.corrcoef(pocos_concatenados, rowvar=False)

    # Create the heatmap with custom labels.
    sns.heatmap(
        corr_matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="coolwarm",
        xticklabels=logs,
        yticklabels=logs,
    )

    plt.title("Correlation Matrix of Real Logs")
    plt.show()

    return


def plot_gan_convergence(history, save_path=None):
    """
    Plot adversarial and physics-informed convergence curves for ChemoGAN.

    Parameters
    ----------
    history : dict
        Training history dictionary returned by the ChemoGAN trainer. Expected
        keys include ``g_loss``, ``d_loss``, ``g_adv``, ``val_g_loss``,
        ``nmr_loss``, ``corr_rhob_nphi``, and ``corr_dt_nphi``. If
        ``pe_ca_loss`` is present, it is also plotted.
    save_path : str or None, optional
        If provided, the figure is saved to this path with 300 dpi.

    Notes
    -----
    This figure is intended for inclusion in the results section of the paper.
    """
    epochs = range(1, len(history["g_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Adversarial Dynamics ---
    axes[0].plot(
        epochs,
        history["d_loss"],
        label="Discriminator Loss",
        color="darkred",
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        history["g_adv"],
        label="Generator Adv Loss",
        color="navy",
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        history["val_g_loss"],
        label="Val Generator Tot",
        color="orange",
        linestyle="dashed",
        alpha=0.8,
    )
    axes[0].set_title("(a) Adversarial Dynamics and Validation", fontsize=14)
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("Loss (Logistic)", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Panel 2: Physical Convergence (Physics-Informed) ---
    axes[1].plot(
        epochs,
        history["nmr_loss"],
        label=r"$\mathcal{L}_{NMR}$ (Porosity Chain)",
        color="forestgreen",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        history["corr_rhob_nphi"],
        label=r"$\mathcal{L}_{Corr}$ ($\rho_b$ vs $\phi_N$)",
        color="purple",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        history["corr_dt_nphi"],
        label=r"$\mathcal{L}_{Corr}$ ($\Delta t$ vs $\phi_N$)",
        color="teal",
        linewidth=2,
    )

    # Add geochemical penalty if available in the history dictionary.
    if "pe_ca_loss" in history:
        axes[1].plot(
            epochs,
            history["pe_ca_loss"],
            label=r"$\mathcal{L}_{Corr}$ (PE vs DWSI)",
            color="crimson",
            linewidth=2,
        )

    axes[1].set_title("(b) Convergence of Physical Constraints", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("Penalty Loss", fontsize=12)
    axes[1].set_yscale("log")  # Log scale helps show constraints approaching zero.
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    plt.show()
