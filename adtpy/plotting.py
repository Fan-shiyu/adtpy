# ## Plots

# ### Venn Diagram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from .projection import Proj

def venn_plot(
    obj,
    area1: float = 1.0,
    area2: float = 1.0,
    cat_cex: float = 1.5,          # category label size (relative)
    cex: float = 1.6,              # subset numbers size (relative)
    col=(0.2, 0.2, 0.2, 0.7),      # outline color with alpha (R: grey(0.2, 0.7))
    fill=("red", "yellow"),        # fill colors for left/right sets
    label: bool = False,
    title: bool = False,
    anno: bool = False,
    **kwargs
):
    """
    Draw a pairwise Venn diagram for a Proj object.

    Parameters mirror the R function:
      - area1, area2: set areas (default 1, 1)
      - cat_cex: category label size (relative)
      - cex: subset text size (relative)
      - col: outline RGBA or color
      - fill: (left_color, right_color)
      - label: show set labels 'data1'/'data2'
      - title: show title + PC subtitle
      - anno: show explanatory annotation
    """
    # type/dimension checks (R: is.proj + n_loadings checks)
    if not isinstance(obj, Proj):
        raise TypeError("input must be the class 'Proj'")
    if obj.n_loadings not in (1, 2, 3):
        raise ValueError("invalid dimensions: n_loadings must be 1, 2, or 3")

    # overlap = similarity (rounded to 2 like R)
    overlap = round(float(obj.similarity), 2)

    plt.figure(figsize=(6, 6))
    v = venn2(
        subsets=(area1, area2, overlap),            # (A, B, A∩B)
        set_labels=("data1", "data2") if label else ("", ""),
        **kwargs
    )

    # colors/fills/edges
    if v.get_patch_by_id("10"):
        v.get_patch_by_id("10").set_color(fill[0])
        v.get_patch_by_id("10").set_alpha(0.6)
        v.get_patch_by_id("10").set_edgecolor(col)
        v.get_patch_by_id("10").set_linewidth(2)
    if v.get_patch_by_id("01"):
        v.get_patch_by_id("01").set_color(fill[1])
        v.get_patch_by_id("01").set_alpha(0.6)
        v.get_patch_by_id("01").set_edgecolor(col)
        v.get_patch_by_id("01").set_linewidth(2)
    if v.get_patch_by_id("11"):
        # intersection inherits edge; leave fill auto so overlap remains legible
        v.get_patch_by_id("11").set_edgecolor(col)
        v.get_patch_by_id("11").set_linewidth(2)

    # font sizes (approximate R's cex scaling)
    cat_fs = 10 * cat_cex
    sub_fs = 10 * cex
    if v.set_labels:
        for lbl in v.set_labels:
            if lbl: lbl.set_fontsize(cat_fs)
    if v.subset_labels:
        for lbl in v.subset_labels:
            if lbl: lbl.set_fontsize(sub_fs)

    # Title + subtitle (PCs used)
    if title:
        main = "Similarity between data1 and data2"
        if obj.n_loadings == 1:
            sub = "--based on PC1"
        elif obj.n_loadings == 2:
            sub = "--based on PC1 and PC2"
        else:
            sub = "--based on PC1, PC2 and PC3"
        plt.title(f"{main}\n{sub}", fontsize=14)

    # Annotation text (bottom note)
    if anno:
        if obj.n_loadings == 1:
            text = (
                "Similarity equals the length of the projection,\n"
                "where the projection is from vector2 (PC1 in data2) to vector1 (PC1 in data1)."
            )
        elif obj.n_loadings == 2:
            text = (
                "Similarity equals the square root of the area of the projected plane,\n"
                "where the projection is from plane2 (PC1 and PC2 in data2) to plane1 (PC1 and PC2 in data1)."
            )
        else:
            text = (
                "Similarity equals the cube root of the volume of the projected cube,\n"
                "where the projection is from cube2 (PC1, PC2, PC3 in data2) to cube1 (PC1, PC2, PC3 in data1)."
            )
        plt.figtext(0.5, 0.02, text, ha="center", va="bottom", fontsize=9)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ### Paired Density Plot




import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def pair_density_plot(obj,
                      cols=('black', 'blue'),
                      lwd=3,
                      xlim=(-1, 1),
                      ylim=(-1, 2),
                      legend=True,
                      box_lwd=0.6,
                      box_col='gray',
                      legend_x=-1,
                      legend_y=2,
                      title='',
                      **kwargs):
    """
    Plot pairwise density plots for PCs from d1 and d2.

    Parameters
    ----------
    obj : Proj
        An object of class Proj.
    cols : tuple of str
        Colors for d1 and d2 lines.
    lwd : float
        Line width.
    xlim : tuple of float
        X-axis limits.
    ylim : tuple of float
        Y-axis limits.
    legend : bool
        Whether to show legends.
    box_lwd : float
        Width of legend box outline.
    box_col : str
        Color of legend box outline.
    legend_x : float
        X-position of legend.
    legend_y : float
        Y-position of legend.
    title : str
        Overall plot title.
    kwargs : dict
        Additional keyword arguments for lines or figure.
    """
    if len(cols) != 2:
        raise ValueError("'cols' must have length 2")
    if not isinstance(title, str):
        raise TypeError("'title' must be a string")

    d1 = obj.d1_PCs
    d2 = obj.d2_PCs
    n = obj.n_loadings

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]  # make it iterable

    for i in range(n):
        ax = axes[i]

        # Estimate density
        kde1 = gaussian_kde(d1[:, i])
        kde2 = gaussian_kde(d2[:, i])
        x_vals = np.linspace(xlim[0], xlim[1], 500)
        y1 = kde1(x_vals)
        y2 = kde2(x_vals)

        # Plot densities
        ax.plot(x_vals, y1, color=cols[0], lw=lwd, **kwargs)
        ax.plot(x_vals, y2, color=cols[1], lw=lwd, **kwargs)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        # Box outline
        for spine in ax.spines.values():
            spine.set_linewidth(box_lwd)
            spine.set_color(box_col)

        # Legend
        if legend:
            ax.legend(
                [f'd1 PC{i+1}', f'd2 PC{i+1}'],
                loc='upper left',
                bbox_to_anchor=(0, 1),
                frameon=True,
                edgecolor=box_col,
                fontsize=10
            )

    # Title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()


# ### Paired Correlation Plot



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def pair_correlation_plot(obj,
                          legend=True,
                          point_col='black',
                          line_col='blue',
                          box_lwd=0.6,
                          box_col='gray',
                          legend_x=0.03,
                          legend_y=0.82,
                          title='',
                          figsize=(12, 4),
                          **kwargs):
    """
    Plot scatter and linear fit for each pair of PCs (from d1 and d2).

    Parameters
    ----------
    obj : Proj
        An object of the Proj class with attributes d1 and d2.
    legend : bool
        Whether to show the legend.
    point_col : str
        Color for the points.
    line_col : str
        Color for the regression line.
    box_lwd : float
        Width of the legend box lines.
    box_col : str
        Color of the legend box.
    legend_x, legend_y : float
        Position of the legend.
    title : str
        Main plot title.
    figsize : tuple
        Size of the overall figure.
    """
    if not isinstance(title, str):
        raise TypeError("'title' must be a string")

    d1 = obj.d1_PCs
    d2 = obj.d2_PCs
    n = obj.n_loadings

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=False, sharey=False)

    if n == 1:
        axes = [axes]  # Ensure iterable

    for i in range(n):
        ax = axes[i]
        x = d2[:, i]
        y = d1[:, i]
        ax.scatter(x, y, color=point_col, s=40, label=f'd1 PC{i+1} & d2 PC{i+1}', **kwargs)

        # Fit and plot regression line
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        ax.plot(x, y_pred, color=line_col, linewidth=3, **kwargs)

        # Correlation
        r, _ = pearsonr(x, y)

        if legend:
            ax.legend(loc='upper left', frameon=True, framealpha=1,
                      edgecolor=box_col, fontsize=10, title_fontsize=11)
            ax.text(legend_x, legend_y, f"cor = {r:.2f}", transform=ax.transAxes,
                    fontsize=10, bbox=dict(facecolor='white', edgecolor=box_col, boxstyle='round'))

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'PC{i+1}', fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout()
    plt.show()


# ### Paired Box Plot



import matplotlib.pyplot as plt
import numpy as np

def pair_boxplot(obj,
                 cols=('green', 'orange'),
                 box_lwd=1,
                 box_col='gray',
                 legend_lty='-', 
                 legend_lwd=2,
                 title='Box Plot',
                 legend_text=('data1', 'data2'),
                 points=True,
                 point_col='black',
                 point_cex=0.8,
                 point_pch='o',
                 legend=True,
                 **kwargs):
    """
    Boxplot comparing PCs of data1 and data2 in a proj object.

    Parameters
    ----------
    obj : Proj
        An object of class Proj.
    cols : tuple of str
        Colors for boxplots (data1, data2).
    box_lwd : float
        Line width for the box.
    box_col : str
        Box color.
    legend_lty : str
        Line type in legend (not used in matplotlib).
    legend_lwd : float
        Line width in legend.
    title : str
        Title for the plot.
    legend_text : tuple of str
        Labels for legend.
    points : bool
        Whether to add jittered points.
    point_col : str
        Color of the points.
    point_cex : float
        Size of the points.
    point_pch : str
        Marker symbol.
    legend : bool
        Whether to display the legend.
    kwargs : dict
        Extra arguments for matplotlib.
    """
    d1 = obj.d1_PCs
    d2 = obj.d2_PCs
    n = obj.n_loadings
    h = np.hstack([d1, d2])

    # Reorder columns for display: PC1_d1, PC1_d2, PC2_d1, PC2_d2, etc.
    h_reordered = []
    for i in range(n):
        h_reordered.append(h[:, i])
        h_reordered.append(h[:, i + n])
    h = np.column_stack(h_reordered)

    # Define box positions: close within PC, gap between PCs
    positions = []
    for i in range(n):
        positions.extend([i * 2 + 1, i * 2 + 1.8])

    box_colors = [cols[i % 2] for i in range(2 * n)]
    y_min, y_max = h.min(), h.max()
    ylim = (y_min, y_max + 0.3)
    yticks = np.round(np.linspace(y_min, y_max, 5), 2)

    fig, ax = plt.subplots(figsize=(1.5 * 2 * n, 5))
    bp = ax.boxplot(h,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False)

    # Make the median lines black
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(box_lwd)  # Optional: use your existing box_lwd for consistency

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(box_lwd)
        patch.set_edgecolor(box_col)

    for whisker in bp['whiskers']:
        whisker.set_linewidth(box_lwd)
    for cap in bp['caps']:
        cap.set_linewidth(box_lwd)

    # Add jittered points
    if points:
        for i, pos in enumerate(positions):
            jitter = np.random.normal(loc=0, scale=0.1, size=h.shape[0])
            ax.scatter(pos + jitter, h[:, i],
                       alpha=0.7,
                       color=point_col,
                       s=point_cex * 20,
                       marker=point_pch,
                       zorder=3)

    # Add x and y labels
    ax.set_xticks([positions[i*2] + 0.5 for i in range(n)])
    ax.set_xticklabels([f'PC{i+1}' for i in range(n)])
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_ylabel("Scores")

    if legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=cols[0], lw=legend_lwd, label=legend_text[0]),
            Line2D([0], [0], color=cols[1], lw=legend_lwd, label=legend_text[1])
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  frameon=True, edgecolor=box_col)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ### Paired Violin Plot



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def pair_vioplot(obj,
                 cols=('#fb8072', '#80b1d3'),
                 box_lwd=1,
                 box_col='gray',
                 legend_lty='-',
                 legend_lwd=2,
                 title='Violin Plot',
                 legend_text=('data1', 'data2'),
                 legend=True,
                 figsize=(10, 6),
                 **kwargs):
    """
    Python version of pair_vioplot for 'proj' class object.
    """
    # Extract data
    d1 = obj.d1_PCs
    d2 = obj.d2_PCs
    n = obj.n_loadings

    # Stack data appropriately
    if n == 3:
        h = np.column_stack((d1[:, 0], d2[:, 0], d1[:, 1], d2[:, 1], d1[:, 2], d2[:, 2]))
    elif n == 2:
        h = np.column_stack((d1[:, 0], d2[:, 0], d1[:, 1], d2[:, 1]))
    else:
        h = np.column_stack((d1[:, 0], d2[:, 0]))

    # Prepare long-form dataframe with hue
    data = []
    group = []
    hue = []
    pc_names = [f"PC{i+1}" for i in range(n)]

    for i in range(n):
        data.extend(h[:, 2*i])      # d1
        data.extend(h[:, 2*i+1])    # d2
        group.extend([pc_names[i]] * h.shape[0])
        group.extend([pc_names[i]] * h.shape[0])
        hue.extend([legend_text[0]] * h.shape[0])
        hue.extend([legend_text[1]] * h.shape[0])

    df = pd.DataFrame({'Scores': data, 'PC': group, 'Dataset': hue})

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x='PC', y='Scores', hue='Dataset', data=df,
                        palette=cols, linewidth=box_lwd, legend=False, **kwargs)

    # Title and axis formatting
    plt.title(title, fontsize=14)
    plt.ylabel('Scores')
    plt.xlabel('')
    plt.xticks(rotation=0)

    # Add manual legend
    if legend:
        handles = [
            plt.Line2D([0], [0], color=cols[0], lw=legend_lwd, label=legend_text[0]),
            plt.Line2D([0], [0], color=cols[1], lw=legend_lwd, label=legend_text[1])
        ]
        ax.legend(handles=handles, frameon=True, edgecolor=box_col)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
