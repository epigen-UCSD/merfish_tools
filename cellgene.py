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


class ScanpyObject:
    def __init__(self, mfx) -> None:
        self.mfx = mfx
        self.cmap = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
                     '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8',
                     '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
        sc.settings.figdir = self.mfx.analysis_folder
        sc.set_figure_params(dpi_save=300, figsize=(5, 5))

    @cached_property
    def scdata(self):
        if not os.path.exists(os.path.join(self.mfx.analysis_folder, 'scanpy_object.h5ad')) or config.get('rerun'):
            return self.initialize()
        else:
            return sc.read(os.path.join(self.mfx.analysis_folder, 'scanpy_object.h5ad'))

    @announce("Building scanpy object and normalizing counts")
    def initialize(self):
        scdata = sc.read_csv(os.path.join(self.mfx.analysis_folder, "single_cell_raw_counts.csv"), first_column_names=True)
        gcells = self.mfx.global_cell_positions#[['global_y', 'global_x']]
        scdata.obsm['X_spatial'] = np.array(gcells.reindex(index=scdata.obs.index.astype(int)))
        sc.pp.calculate_qc_metrics(scdata, percent_top=None, inplace=True)
        sc.pp.normalize_total(scdata, target_sum=np.median(scdata.obs['total_counts']))
        sc.pp.log1p(scdata)
        scdata.raw = scdata
        scdata.X = scdata.to_df().apply(zscore, axis=0).to_numpy()
        scdata.write(os.path.join(self.mfx.analysis_folder, 'scanpy_object.h5ad'))
        return scdata

    @cached_property
    def number_PCs(self):
        def jumble(seq):
            return sample(list(seq), k=len(seq))

        mvars = []
        for i in tqdm(range(20), desc="Optimizing number of PCs"):
            randomized = self.scdata.to_df().apply(jumble, axis=0)
            rndata = AnnData(randomized)
            sc.tl.pca(rndata, svd_solver='arpack')
            mvars.append(rndata.uns['pca']['variance'][0])

        cutoff = np.mean(mvars)
        sc.tl.pca(self.scdata, svd_solver='arpack')
        return int(np.sum(self.scdata.uns['pca']['variance'] > cutoff))

    def cluster_cells(self):
        print("Calculating neighbors...", end='')
        sc.pp.neighbors(self.scdata, n_pcs=self.number_PCs)
        print("done\nLouvain clustering...", end='')
        sc.tl.louvain(self.scdata)
        print("done\nLeiden clustering...", end='')
        sc.tl.leiden(self.scdata)
        print("done\nCalculating UMAP...", end='')
        sc.tl.paga(self.scdata)
        sc.pl.paga(self.scdata, plot=False)
        sc.tl.umap(self.scdata, init_pos='paga')
        nclusts = len(np.unique(self.scdata.obs['leiden']))
        sc.pl.umap(self.scdata, color='leiden', add_outline=True, legend_loc='on data',
                   legend_fontsize=12, legend_fontoutline=2, frameon=False,
                   title=f'{nclusts} clusters of {len(self.scdata):,d} cells', palette=self.cmap, save='_clusters.png')
        self.scdata.write(os.path.join(self.mfx.analysis_folder, 'scanpy_object.h5ad'))
        print("done")

    @cached_property
    def nclusts(self):
        if 'leiden' not in self.scdata.obs:
            self.cluster_cells()
        return len(np.unique(self.scdata.obs['leiden']))
