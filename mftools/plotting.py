import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import zscore, pearsonr
from matplotlib.ticker import FuncFormatter
from skimage import transform

from . import config
from . import stats
from . import util

pctformatter = FuncFormatter(lambda x, pos: f"{x*100:.1f}%")


def plot(save: str, figsize=None, style: str = "default"):
    def decorator_plot(func):
        def plot_wrapper(*args, **kwargs):
            print(f"Creating plot {save}")
            plt.style.use(style)
            plt.figure(figsize=figsize)
            rval = func(*args, **kwargs)
            plt.tight_layout()
            plt.savefig(config.path(save), dpi=300)
            return rval

        return plot_wrapper

    return decorator_plot


def drift_histogram(mfx) -> None:
    # Remove outliers
    drifts = mfx.hyb_drifts[(np.abs(zscore(mfx.hyb_drifts)) < 3).all(axis=1)]

    # This is for setting the min and max on the axes
    driftlist = list(drifts["X drift"]) + list(drifts["Y drift"])

    g = sns.FacetGrid(
        drifts,
        col="Hybridization round",
        col_wrap=3,
        xlim=(min(driftlist), max(driftlist)),
        ylim=(min(driftlist), max(driftlist)),
    )
    g.map_dataframe(sns.histplot, x="X drift", y="Y drift", cmap="viridis")
    g.set(ylabel="Y drift", xlabel="X drift")
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("drift_histogram.png"), dpi=300)


@plot(save="exact_vs_corrected.png")
def exact_vs_corrected() -> None:
    exact = stats.get("Exact barcode count") / 1_000_000
    corrected = stats.get("Corrected barcode count") / 1_000_000
    plt.bar([0, 1], [exact, corrected])
    plt.xticks([0, 1], ["Exact", "Corrected"])
    plt.ylabel("Counts (x10^6)")
    plt.text(0, exact, f"{stats.get('% exact barcodes')*100:0.0f}%", ha="center", va="bottom")
    plt.text(
        1,
        corrected,
        f"{(stats.get('0->1 error rate') + stats.get('1->0 error rate'))*100:0.0f}%",
        ha="center",
        va="bottom",
    )
    plt.grid(b=False)


@plot(save="confidence_ratio.png")
def confidence_ratios(per_gene_error) -> None:
    gene_stats = per_gene_error.sort_values(by="% exact barcodes", ascending=False)
    colors = [
        "steelblue" if "blank" not in name and "notarget" not in name else "firebrick" for name in gene_stats.index
    ]
    plt.bar(
        range(1, len(gene_stats) + 1),
        gene_stats["% exact barcodes"],
        width=1,
        color=colors,
    )
    plt.ylim(
        [
            np.min(gene_stats["% exact barcodes"]) * 0.95,
            np.max(gene_stats["% exact barcodes"]) * 1.05,
        ]
    )
    plt.xlim(1, len(gene_stats))
    plt.xlabel("Rank")
    plt.ylabel("Confidence ratio")
    plt.grid(b=False)


def per_bit_error_bar(per_bit_error, colors) -> None:
    ax = sns.catplot(
        x="Hybridization round",
        y="Error rate",
        row="Error type",
        data=per_bit_error,
        kind="bar",
        height=2,
        aspect=3,
        capsize=0.2,
        hue="Color",
        order=sorted(per_bit_error["Hybridization round"].unique()),
        hue_order=colors,
        sharex=False,
        sharey=False,
    )
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    ax.savefig(config.path("bit_error_bar.png"), dpi=300)


def per_bit_error_line(per_bit_error, colors) -> None:
    ax = sns.catplot(
        x="Hybridization round",
        y="Error rate",
        row="Error type",
        data=per_bit_error,
        kind="point",
        height=2,
        aspect=3,
        ci=None,
        capsize=0.2,
        hue="Color",
        order=sorted(per_bit_error["Hybridization round"].unique()),
        hue_order=colors,
        sharey=False,
        sharex=False,
    )
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    ax.savefig(config.path("bit_error_line.png"), dpi=300)


def per_hyb_error(per_bit_error) -> None:
    g = sns.FacetGrid(per_bit_error, row="Error type", height=2, aspect=3, sharey=False, sharex=False)
    g.map_dataframe(sns.barplot, x="Hybridization round", y="Error rate", ci=None)
    g.map_dataframe(
        sns.pointplot,
        x="Hybridization round",
        y="Error rate",
        color="#444444",
        markers=".",
        capsize=0.2,
    )
    for axs in g.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    g.set(ylabel="Error rate")
    g.axes.flat[-1].set_xlabel("Hybridization round")
    plt.tight_layout()
    g.savefig(config.path("hyb_error.png"), dpi=300)


def per_color_error(per_bit_error, colors) -> None:
    ax = sns.catplot(
        x="Color",
        y="Error rate",
        row="Error type",
        data=per_bit_error,
        kind="bar",
        height=2,
        aspect=3,
        capsize=0.2,
        order=colors,
        sharey=False,
        sharex=False,
    )
    for axs in ax.axes.flat:
        axs.yaxis.set_major_formatter(pctformatter)
        axs.grid(b=False)
    plt.tight_layout()
    ax.savefig(config.path("color_error.png"), dpi=300)


def fov_error_bar(per_fov_error) -> None:
    def fovplot(d, ax):
        q1 = d["Error rate"].quantile(0.25)
        q3 = d["Error rate"].quantile(0.75)
        outlier = q3 + 1.5 * (q3 - q1)
        ax.bar(d["FOV"], d["Error rate"], width=1)
        ax.axhline(outlier, color="red", linestyle=":")
        for i, row in d[d["Error rate"] >= outlier].iterrows():
            ax.annotate(
                row["FOV"],
                (row["FOV"], row["Error rate"]),
                ha="center",
                color="orangered",
                weight="heavy",
            )
        ax.set_xticks(list(range(0, d["FOV"].max(), 25)) + [d["FOV"].max()])
        ax.set_xlabel("FOV")
        ax.set_ylabel("Error rate")
        ax.set_ylim(bottom=d["Error rate"].min() * 0.75)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    fovplot(per_fov_error[per_fov_error["Error type"] == "1->0"], ax[0])
    fovplot(per_fov_error[per_fov_error["Error type"] == "0->1"], ax[1])
    ax[0].set_title("1->0 error rate")
    ax[1].set_title("0->1 error rate")
    plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(config.path("fov_error.png"), dpi=300)


def fov_error_spatial(per_fov_error, positions) -> None:
    fovdata = pd.merge(per_fov_error, positions, left_on="FOV", right_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    sns.scatterplot(
        ax=ax1,
        data=fovdata[fovdata["Error type"] == "1->0"],
        x="x",
        y="y",
        hue="Error rate",
        palette="YlOrRd",
    )
    sns.scatterplot(
        ax=ax2,
        data=fovdata[fovdata["Error type"] == "0->1"],
        x="x",
        y="y",
        hue="Error rate",
        palette="YlOrRd",
    )
    ax1.axis("off")
    ax1.set_title("1->0 error rate")
    ax2.axis("off")
    ax2.set_title("0->1 error rate")
    fig.tight_layout()
    fig.savefig(config.path("fov_error_spatial.png"), dpi=300)


def rnaseq_correlation(bcs, dataset) -> None:
    refcounts = util.reference_gene_counts(dataset["file"])
    plot_correlation(
        xcounts=refcounts,
        ycounts=np.log10(bcs["gene"].value_counts()),
        xlabel=f"Log {dataset['name']} Counts",
        ylabel="Log MERFISH Counts",
        outfile=config.path(f"correlation_{dataset['name']}.png"),
        omit=[],
    )


def transcripts_per_gene_scatterplot(x_adata, y_adata, xlabel=None, ylabel=None, gene_labels=True):
    common_genes = x_adata.var_names.intersection(y_adata.var_names)

    x_totals = x_adata[:, common_genes].X.sum(axis=0)
    y_totals = y_adata[:, common_genes].X.sum(axis=0)

    corr, _ = pearsonr(x_totals, y_totals)
    trendline = np.poly1d(np.polyfit(x_totals, y_totals, 1))
    plt.figure(figsize=(10, 7))
    plt.plot((min(x_totals), max(x_totals)), (trendline(min(x_totals)), trendline(max(x_totals))), "r:")
    plt.text(0.05, 0.95, f"p={corr:.2f}", transform=plt.gca().transAxes, va="top")
    plt.scatter(x_totals, y_totals)
    if gene_labels:
        for i, txt in enumerate(common_genes):
            plt.annotate(txt, (x_totals[i], y_totals[i]))
    if not xlabel and "dataset_name" in x_adata.uns:
        xlabel = x_adata.uns["dataset_name"]
    if not ylabel and "dataset_name" in y_adata.uns:
        ylabel = y_adata.uns["dataset_name"]
    plt.ylabel(f"Total transcripts ({ylabel})")
    plt.xlabel(f"Total transcripts ({xlabel})")
    plt.tight_layout()


@plot(save="cell_volume.png")
def cell_volume_histogram(celldata):
    sns.histplot(celldata["volume"], bins=50)
    plt.xlabel("Cell volume (pixels)")


def plot_mask(mask):
    plt.figure()
    levels = np.unique(mask) + 0.5
    plt.contour(mask, sorted(levels), c="r")
    for cellid in np.unique(mask):
        if cellid > 0:
            plt.text(np.median(mask[mask == cellid]), s=cellid)


@plot(save="transcript_count_per_cell.png", figsize=(7, 5))
def counts_per_cell_histogram(counts):
    sns.histplot(counts.apply(np.sum, axis=1), bins=50)
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.grid(b=False)


@plot(save="genes_detected_per_cell.png", figsize=(7, 5))
def genes_detected_per_cell_histogram(counts):
    sns.histplot(counts.astype(bool).sum(axis=1), binwidth=1)
    plt.xlabel("Genes detected")
    plt.ylabel("Cell count")
    plt.grid(b=False)


@plot(save="spatial_transcripts_per_fov.png", figsize=(8, 10))
def spatial_transcripts_per_fov(bcs, positions):
    plt.scatter(
        positions["y"],
        -positions["x"],
        c=bcs.groupby("fov").count()["gene"],
        cmap="jet",
    )
    plt.colorbar()


@plot(save="spatial_cell_clusters.png", style="dark_background")
def spatial_cell_clusters(adata):
    sc.pl.embedding(adata, basis="X_spatial", color="leiden")


@plot(save="umap_clusters.png")
def umap_clusters(adata):
    nclusts = len(np.unique(adata.obs["leiden"]))
    sc.pl.umap(
        adata,
        color="leiden",
        add_outline=True,
        legend_loc="on data",
        legend_fontsize=12,
        legend_fontoutline=2,
        frameon=False,
        title=f"{nclusts} clusters of {len(adata):,d} cells",
        # palette=self.cmap,
        # save="_clusters.png",
    )


def fov_number_map(mfx):
    plt.figure(figsize=(12, 16), facecolor="white")
    plt.scatter(x=mfx.positions["x"], y=mfx.positions["y"], c="w")
    for i, row in mfx.positions.iterrows():
        plt.text(x=row["x"], y=row["y"], s=row.name)
    plt.grid(b=False)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(config.path("fov_number_map.png"), dpi=150)


def check_drift(images, merlin, fov, bit1, bit2):
    drifts = merlin.load_drift_transformations(fov=fov)
    drift1 = drifts[bit1 - 1]
    drift2 = drifts[bit2 - 1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
    fig.tight_layout()
    img1 = images.load_image(fov=fov, channel=f"bit{bit1}", fiducial=True)
    img2 = images.load_image(fov=fov, channel=f"bit{bit2}", fiducial=True)
    comb1 = create_color_image(red=img1, blue=img2, vmax=99)
    axes[0].imshow(comb1)
    axes[0].axis("off")
    axes[0].set_title("Raw fiducial images")
    axes[0].text(0.02, 0.98, s=f"Bit {bit1}", c="r", transform=axes[0].transAxes, va="top")
    axes[0].text(0.02, 0.94, s=f"Bit {bit2}", c="b", transform=axes[0].transAxes, va="top")
    img1 = transform.warp(img1, drift1, preserve_range=True).astype(img1.dtype)
    img2 = transform.warp(img2, drift2, preserve_range=True).astype(img2.dtype)
    comb2 = create_color_image(red=img1, blue=img2, vmax=99)
    axes[1].imshow(comb2)
    axes[1].axis("off")
    axes[1].set_title("Aligned fiducial images")
    axes[1].text(
        0.02,
        0.98,
        s=f"{drift1.params[0][2]:0.2f}, {drift1.params[1][2]:0.2f}",
        c="r",
        transform=axes[1].transAxes,
        va="top",
    )
    axes[1].text(
        0.02,
        0.94,
        s=f"{drift2.params[0][2]:0.2f}, {drift2.params[1][2]:0.2f}",
        c="b",
        transform=axes[1].transAxes,
        va="top",
    )
    img1 = images.load_image(fov=fov, channel=f"bit{bit1}", max_projection=True)
    img2 = images.load_image(fov=fov, channel=f"bit{bit2}", max_projection=True)
    img1 = transform.warp(img1, drift1, preserve_range=True).astype(img1.dtype)
    img2 = transform.warp(img2, drift2, preserve_range=True).astype(img2.dtype)
    comb3 = create_color_image(red=img1, blue=img2, vmax=99)
    axes[2].imshow(comb3)
    axes[2].axis("off")
    axes[2].set_title("Aligned bit images")


def create_color_image(
    red: np.ndarray = None, green: np.ndarray = None, blue: np.ndarray = None, vmax=100
) -> np.ndarray:
    shape = [x for x in [red, green, blue] if x is not None][0].shape
    if red is not None:
        red = red / np.percentile(red, vmax)
        red[red > 1] = 1
    else:
        red = np.zeros(shape)
    if green is not None:
        green = green / np.percentile(green, vmax)
        green[green > 1] = 1
    else:
        green = np.zeros(shape)
    if blue is not None:
        blue = blue / np.percentile(blue, vmax)
        blue[blue > 1] = 1
    else:
        blue = np.zeros(shape)
    img = np.array([red, green, blue])
    img = np.moveaxis(img, 0, -1)
    return img
