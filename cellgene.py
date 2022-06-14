import os
from random import sample
from functools import cached_property

import scanpy as sc
import numpy as np
from scipy.stats import zscore
from anndata import AnnData
from tqdm import tqdm

from util import announce
import config
import stats


def create_scanpy_object(cellgene, celldata):
    adata = sc.AnnData(cellgene, dtype=np.int32)
    adata.obsm["X_spatial"] = np.array(
        celldata[["global_x", "global_y"]].reindex(index=adata.obs.index.astype(int))
    )
    sc.pp.filter_cells(adata, min_genes=3)
    # self.mfx.update_filtered_celldata("Low genes")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=10)
    sc.pp.normalize_total(adata)
    adata.raw = adata
    sc.pp.scale(adata)
    return adata


def optimize_number_PCs(adata):
    def jumble(seq):
        return sample(list(seq), k=len(seq))

    mvars = []
    for i in tqdm(range(20), desc="Optimizing number of PCs"):
        randomized = adata.to_df().apply(jumble, axis=0)
        rndata = AnnData(randomized, dtype=adata.X.dtype)
        sc.tl.pca(rndata, svd_solver="arpack")
        mvars.append(rndata.uns["pca"]["variance"][0])

    cutoff = np.mean(mvars)
    sc.tl.pca(adata, svd_solver="arpack")
    n_pcs = int(np.sum(adata.uns["pca"]["variance"] > cutoff))
    stats.set("Number of PCs", n_pcs)
    return n_pcs


def cluster_cells(adata, n_pcs):
    if "pca" not in adata.uns:
        sc.tl.pca(adata, svd_solver="arpack")
    print("Calculating neighbors...", end="")
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    print("done\nLeiden clustering...", end="")
    sc.tl.leiden(adata)
    print("done\nCalculating UMAP...", end="")
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga", min_dist=0.3, spread=1)
    print("done")
    stats.set("Number of clusters", len(np.unique(adata.obs["leiden"])))
