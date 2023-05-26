import os
from random import sample
from functools import cached_property

import scanpy as sc
import numpy as np
from anndata import AnnData
from tqdm import tqdm
from scipy.stats import pearsonr, zscore
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd

from . import stats


def create_scanpy_object(analysis, name=None, positions=None, codebook=None, keep_empty_cells=True) -> sc.AnnData:
    cellgene = analysis.load_cell_by_gene_table()
    celldata = analysis.load_cell_metadata()
    celldata.index = celldata.index.astype(str)
    if keep_empty_cells:
        empty_cells = pd.DataFrame(
            [
                pd.Series(data=0, index=cellgene.columns, name=cellid)
                for cellid in celldata.index.difference(cellgene.index)
            ]
        )
        cellgene = pd.concat([cellgene, empty_cells])
    blank_cols = np.array(["notarget" in col or "blank" in col.lower() for col in cellgene])
    adata = sc.AnnData(cellgene.loc[:, ~blank_cols], dtype=np.int32)
    adata.obsm["X_blanks"] = cellgene.loc[:, blank_cols].to_numpy()
    adata.uns["blank_names"] = cellgene.columns[blank_cols].to_list()
    if "global_x" in celldata:
        adata.obsm["X_spatial"] = np.array(
            celldata[["global_x", "global_y"]].reindex(index=adata.obs.index)
        )
    elif "center_x" in celldata:
        adata.obsm["X_spatial"] = np.array(
            celldata[["center_x", "center_y"]].reindex(index=adata.obs.index)
        )
    if "fov_x" in celldata:
        adata.obsm["X_local"] = np.array(celldata[["fov_x", "fov_y"]].reindex(index=adata.obs.index))
    for column in celldata.columns:
        adata.obs[column] = celldata[column]
    adata.obs["fov"] = adata.obs["fov"].astype(str)
    adata.layers["counts"] = adata.X
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    adata.obs["blank_counts"] = adata.obsm["X_blanks"].sum(axis=1)
    adata.obs["misid_rate"] = (adata.obs["blank_counts"] / len(adata.uns["blank_names"])) / (
        adata.obs["total_counts"] / len(adata.var_names)
    )
    adata.obs["counts_per_volume"] = adata.obs["total_counts"] / adata.obs["volume"]
    if codebook:
        adata.varm["codebook"] = codebook.set_index("name").loc[adata.var_names].filter(like="bit").to_numpy()
        for bit in range(adata.varm["codebook"].shape[1]):
            adata.obs[f"bit{bit+1}"] = adata[:, adata.varm["codebook"][:, bit] == 1].X.sum(axis=1)
    if positions:
        adata.uns["fov_positions"] = positions.to_numpy()
    if name:
        adata.uns["dataset_name"] = name
    else:
        adata.uns["dataset_name"] = analysis.root.name
    return adata


def adjust_spatial_coordinates(adata, flip_horizontal=False, flip_vertical=False, transpose=False):
    if transpose and (flip_horizontal or flip_vertical):
        pass  # TODO: Should warn about order of operations
    if transpose:
        adata.obsm["X_spatial"] = np.roll(adata.obsm["X_spatial"], 1, 1)
    if flip_horizontal:
        adata.obsm["X_spatial"][:, 0] = -adata.obsm["X_spatial"][:, 0]
    if flip_vertical:
        adata.obsm["X_spatial"][:, 1] = -adata.obsm["X_spatial"][:, 1]


def normalize(adata, scale=False):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=2)
    adata.raw = adata
    if scale:
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


def find_cell_communities(adata: sc.AnnData, labels: str, radius: int = 150) -> None:
    """Group cells based on the cell types present in their vicinity."""
    ngraph = NearestNeighbors(radius=radius)
    ngraph.fit(adata.obsm["X_spatial"])
    _, indexes = ngraph.radius_neighbors(adata.obsm["X_spatial"])
    res = [pd.DataFrame(adata[cells, :].obs[labels].value_counts()) for cells in tqdm(indexes)]
    neighbors_table = pd.concat(res, axis=1)
    neighbors_table.columns = adata.obs_names
    neighbors_table = neighbors_table.fillna(0)
    adata.obsm["X_neighborhood"] = (neighbors_table / neighbors_table.sum(axis=0)).T.to_numpy()
    kmeans = KMeans().fit(adata.obsm["X_neighborhood"])
    adata.obs["community"] = kmeans.labels_.astype(str)


def transfer_labels(adata: sc.AnnData, refdata: sc.AnnData, label: str, **kwargs):
    common_genes = adata.var_names.intersection(refdata.var_names)
    classifier = KNeighborsClassifier(**kwargs)  # n_neighbors=5, n_jobs=24, metric='correlation')
    classifier.fit(refdata[:, common_genes].X, refdata.obs[label])
    return classifier.predict(adata[:, common_genes].X)


def label_clusters(adata: sc.AnnData, refdata: sc.AnnData, label: str, ref_label: str, number_sep=None):
    common_genes = adata.var_names.intersection(refdata.var_names)
    df = adata[:, common_genes].to_df()
    df["cluster"] = adata.obs[label]
    df = df.groupby("cluster").mean()
    df = df.apply(zscore, axis=0)
    df = df.dropna(axis=1)
    refdf = refdata[:, common_genes].to_df()
    refdf["cluster"] = refdata.obs[ref_label]
    refdf = refdf.groupby("cluster").mean()
    refdf = refdf.apply(zscore, axis=0)
    refdf = refdf.dropna(axis=1)
    common_genes = refdf.columns.intersection(df.columns)
    df = df[common_genes]
    refdf = refdf[common_genes]
    # Get all the correlations
    ps = []
    for _, row1 in df.iterrows():
        ps_ = []
        for _, row2 in refdf.iterrows():
            ps_.append(pearsonr(row1, row2)[0])
        ps.append(ps_)
    cordf = pd.DataFrame(ps, index=df.index, columns=refdf.index)
    mapping = cordf.idxmax(axis=1)
    if number_sep is not None:
        counts = mapping.value_counts()
        counter = {k: 1 for k in counts[counts > 1].index}
        for index, name in enumerate(mapping):
            if name in counter:
                mapping[index] = name + number_sep + str(counter[name])
                counter[name] += 1
    return mapping.loc[adata.obs[label]].values
