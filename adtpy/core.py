# ## Data Simulation


import numpy as np
import pandas as pd

def simulation(data: pd.DataFrame, mean: float = 0, sd: float = 1, seed: int = 2023) -> pd.DataFrame:
    """
    Create a simulated dataset similar to the input by adding Gaussian noise.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset (numeric).
    mean : float, default=0
        Mean of the Gaussian noise.
    sd : float
        Standard deviation of the Gaussian noise.
    seed : int, default=2023
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Simulated dataset with the same shape as `data`.
    """
    np.random.seed(seed)
    noise = np.random.normal(loc=mean, scale=sd, size=data.shape)
    return data + noise


# ## Normalization and PCA

import numpy as np
import pandas as pd

def scree_plot(vari_explain):
    """Helper: scree plot of cumulative variance explained."""
    x = np.arange(1, len(vari_explain) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(x, vari_explain, marker="o", color="black")
    plt.xticks(x)
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Proportion of Variance Explained")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()





from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def get_loadings(data, explain=0.9, n_loadings="auto", normalize=True, scree_plot_flag=False):
    """
    Select PCA loadings based on cumulative variance explained.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input dataset (numeric).
    explain : float, default=0.9
        Target cumulative proportion of variance explained.
    n_loadings : 'auto' or int (1–3), default='auto'
        How many loadings to return.
    normalize : bool, default=True
        Whether to normalize features to [0,1].
    scree_plot_flag : bool, default=False
        Whether to plot cumulative variance explained.

    Returns
    -------
    np.ndarray
        Selected loadings matrix (shape: n_features x n_components).
    """
    # checks
    if not isinstance(explain, (int, float)):
        raise ValueError('argument "explain" must be numeric')
    if n_loadings != "auto" and (n_loadings < 1 or n_loadings > 3):
        raise ValueError('argument "n_loadings" must be between 1 and 3')

    # ensure DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # normalization (range scaling like caret::preProcess(method="range"))
    X = data.values
    if normalize:
        X = MinMaxScaler().fit_transform(X)

    # PCA via eigen decomposition
    pca = PCA(svd_solver="full")  # full eigen decomposition
    pca.fit(X)

    # PCA loadings (eigenvectors scaled by eigenvalues)
    # sklearn: components_.T gives loadings
    loadings = -pca.components_.T   # match R's sign convention

    # cumulative proportion of variance explained
    vari_explain = np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_)

    # selecting loadings
    if n_loadings == "auto":
        if vari_explain[0] >= explain:
            loadings = loadings[:, [0]]
        elif vari_explain[1] >= explain:
            loadings = loadings[:, [0, 1]]
        else:
            loadings = loadings[:, [0, 1, 2]]
    else:
        n_loadings = round(n_loadings)
        loadings = loadings[:, :n_loadings]

    # scree plot
    if scree_plot_flag:
        scree_plot(vari_explain)

    return loadings