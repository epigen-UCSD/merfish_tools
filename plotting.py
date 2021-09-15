import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from matplotlib.ticker import FuncFormatter

import config


pctformatter = FuncFormatter(lambda x, pos: f"{x*100:.1f}%")


def drift_histogram(mfx) -> None:
    # Remove outliers
    drifts = mfx.hyb_drifts[(np.abs(zscore(mfx.hyb_drifts)) < 3).all(axis=1)]

    # This is for setting the min and max on the axes
    driftlist = list(drifts['X drift']) + list(drifts['Y drift'])

    g = sns.FacetGrid(drifts, col="Hybridization round", col_wrap=3,
                      xlim=(min(driftlist), max(driftlist)), ylim=(min(driftlist), max(driftlist)))
    g.map_dataframe(sns.histplot, x="X drift", y="Y drift", cmap="viridis")
    g.set(ylabel="Y drift", xlabel="X drift")
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("drift_histogram.png"), dpi=300)


def exact_vs_corrected(mfx: 'MerfishExperiment') -> None:
    exact = mfx.stats['Exact barcode count'] / 1_000_000
    corrected = mfx.stats['Corrected barcode count'] / 1_000_000
    plt.figure()
    plt.bar([0, 1], [exact, corrected])
    plt.xticks([0, 1], ['Exact', 'Corrected'])
    plt.ylabel('Counts (x10^6)')
    plt.text(0, exact, f"{mfx.stats['% exact barcodes']*100:0.0f}%", ha='center', va='bottom')
    plt.text(1, corrected, f"{(mfx.stats['0->1 error rate'] + mfx.stats['1->0 error rate'])*100:0.0f}%", ha='center', va='bottom')
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("exact_vs_corrected.png"), dpi=300)


def confidence_ratios(mfx: 'MerfishExperiment') -> None:
    gene_stats = mfx.stats.per_gene_error.sort_values(by='% exact barcodes', ascending=False)
    colors = ['steelblue' if 'blank' not in name and 'notarget' not in name else 'firebrick' for name in gene_stats.index]
    plt.figure()
    plt.bar(range(1, len(gene_stats)+1), gene_stats['% exact barcodes'], width=1, color=colors)
    plt.ylim([np.min(gene_stats['% exact barcodes'])*0.95, np.max(gene_stats['% exact barcodes'])*1.05])
    plt.xlim(1, len(gene_stats))
    plt.xlabel('Rank')
    plt.ylabel('Confidence ratio')
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("confidence_ratio.png"), dpi=300)


def per_bit_error_bar(mfx: 'MerfishExperiment') -> None:
    data = mfx.stats.per_bit_error

    ax = sns.catplot(x="Hybridization round", y="Error rate", row="Error type", data=data,
                     kind="bar", height=2, aspect=3, capsize=0.2, hue='Color',
                     order=sorted(data["Hybridization round"].unique()),
                     hue_order=mfx.barcode_colors, sharex=False, sharey=False)
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    ax.savefig(config.path("bit_error_bar.png"), dpi=300)


def per_bit_error_line(mfx: 'MerfishExperiment') -> None:
    data = mfx.stats.per_bit_error

    ax = sns.catplot(x="Hybridization round", y="Error rate", row="Error type", data=data,
                     kind="point", height=2, aspect=3, ci=None, capsize=0.2, hue='Color',
                     order=sorted(data["Hybridization round"].unique()),
                     hue_order=mfx.barcode_colors, sharey=False, sharex=False)
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    ax.savefig(config.path("bit_error_line.png"), dpi=300)


def per_hyb_error(mfx: 'MerfishExperiment') -> None:
    data = mfx.stats.per_bit_error

    g = sns.FacetGrid(data, row="Error type", height=2, aspect=3, sharey=False, sharex=False)
    g.map_dataframe(sns.barplot, x="Hybridization round", y="Error rate", ci=None)
    g.map_dataframe(sns.pointplot, x="Hybridization round", y="Error rate", color='#444444',
                    markers='.', capsize=0.2)
    for axs in g.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    g.set(ylabel="Error rate")
    g.axes.flat[-1].set_xlabel("Hybridization round")
    plt.tight_layout()
    g.savefig(config.path("hyb_error.png"), dpi=300)


def per_color_error(mfx: 'MerfishExperiment') -> None:
    data = mfx.stats.per_bit_error

    ax = sns.catplot(x="Color", y="Error rate", row="Error type", data=data, kind="bar",
                     height=2, aspect=3, capsize=0.2, order=mfx.barcode_colors,
                     sharey=False, sharex=False)
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    plt.tight_layout()
    ax.savefig(config.path("color_error.png"), dpi=300)


def fov_error_bar(mfx: 'MerfishExperiment') -> None:
    def fovplot(d, ax):
        q1 = d['Error rate'].quantile(0.25)
        q3 = d['Error rate'].quantile(0.75)
        outlier = q3 + 1.5 * (q3 - q1)
        ax.bar(d['FOV'], d['Error rate'], width=1)
        ax.axhline(outlier, color='red', linestyle=':')
        for i, row in d[d['Error rate'] >= outlier].iterrows():
            ax.annotate(row['FOV'], (row['FOV'], row['Error rate']), ha='center', color='orangered', weight='heavy')
        ax.set_xticks(list(range(0, d['FOV'].max(), 25)) + [d['FOV'].max()])
        ax.set_xlabel("FOV")
        ax.set_ylabel("Error rate")
        ax.set_ylim(bottom=d['Error rate'].min()*0.75)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    fovplot(mfx.stats.per_fov_error[mfx.stats.per_fov_error['Error type'] == '1->0'], ax[0])
    fovplot(mfx.stats.per_fov_error[mfx.stats.per_fov_error['Error type'] == '0->1'], ax[1])
    ax[0].set_title("1->0 error rate")
    ax[1].set_title("0->1 error rate")
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("fov_error.png"), dpi=300)


def fov_error_spatial(mfx: 'MerfishExperiment') -> None:
    fovdata = pd.merge(mfx.stats.per_fov_error, mfx.positions, left_on="FOV", right_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    sns.scatterplot(ax=ax1, data=fovdata[fovdata['Error type'] == '1->0'], x='x', y='y', hue="Error rate", palette="YlOrRd")
    sns.scatterplot(ax=ax2, data=fovdata[fovdata['Error type'] == '0->1'], x='x', y='y', hue="Error rate", palette="YlOrRd")
    ax1.axis("off")
    ax1.set_title("1->0 error rate")
    ax2.axis("off")
    ax2.set_title("0->1 error rate")
    fig.tight_layout()
    fig.savefig(config.path("fov_error_spatial.png"), dpi=300)


def rnaseq_correlation(mfx: 'MerfishExperiment', dataset) -> None:
    plot_correlation(xcounts=mfx.stats.reference_gene_counts(dataset),
                     ycounts=mfx.stats.merfish_gene_counts,
                     xlabel=f'Log {dataset} Counts',
                     ylabel='Log MERFISH Counts',
                     outfile=config.path(f'correlation_{dataset}.png'),
                     omit=[])


def plot_correlation(xcounts, ycounts, xlabel, ylabel, outfile=None, omit=[]):
    set1 = set(xcounts.keys())
    set2 = set(ycounts.keys())
    genes_to_consider = [gene for gene in list(set1.intersection(set2)) if gene not in omit]

    x = [xcounts[gene] for gene in genes_to_consider]
    y = [ycounts[gene] for gene in genes_to_consider]

    corr, pval = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.figure(figsize=(10, 7))
    plt.plot(x, p(x), "r--")
    plt.scatter(x, y)
    for i, txt in enumerate(genes_to_consider):
        plt.annotate(txt, (x[i], y[i]))
    plt.title("Pearson = %.3f" % corr)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)


def cell_volume_histogram(mfx):
    plt.figure()
    sns.histplot(mfx.celldata['volume'], bins=50)
    plt.xlabel("Cell volume (pixels)")
    plt.tight_layout()
    plt.savefig(config.path("cell_volume.png"), dpi=300)


def plot_mask(mask):
    plt.figure()
    levels = np.unique(mask) + 0.5
    plt.contour(mask, sorted(levels), c='r')
    for cellid in np.unique(mask):
        if cellid > 0:
            plt.text(np.median(mask[mask == cellid]), s=cellid)


def counts_per_cell_histogram(mfx):
    plt.figure(figsize=(7,5))
    sns.histplot(mfx.single_cell_raw_counts.apply(np.sum, axis=1), bins=50)
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.tight_layout()
    plt.grid(b=False)
    plt.savefig(config.path("transcript_count_per_cell.png"), dpi=300)


def genes_detected_per_cell_histogram(mfx):
    plt.figure(figsize=(7,5))
    sns.histplot(mfx.single_cell_raw_counts.astype(bool).sum(axis=1), binwidth=1)
    plt.xlabel("Genes detected")
    plt.ylabel("Cell count")
    plt.tight_layout()
    plt.grid(b=False)
    plt.savefig(config.path("genes_detected_per_cell.png"), dpi=300)


def spatial_cell_clusters(mfx):
    plt.figure(figsize=(10,12), facecolor='black')
    for cluster in np.unique(mfx.clustering.scdata.obs['leiden']):
        inds = mfx.clustering.scdata.obs[mfx.clustering.scdata.obs['leiden'] == cluster].index.astype(int)
        x = mfx.celldata.loc[inds]['global_x']
        y = mfx.celldata.loc[inds]['global_y']
        plt.scatter(y, x, c=mfx.clustering.cmap[int(cluster) % 20], s=0.5)
    plt.grid(b=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(config.path('spatial_cell_clusters.png'), dpi=300)


def fov_number_map(mfx):
    plt.figure(figsize=(12,16), facecolor='white')
    plt.scatter(x=mfx.positions['x'], y=mfx.positions['y'], c='w')
    for i, row in mfx.positions.iterrows():
        plt.text(x=row['x'], y=row['y'], s=row.name)
    plt.grid(b=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(config.path('fov_number_map.png'), dpi=150)
