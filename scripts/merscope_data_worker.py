import pathlib
import logging
import json
from mftools.fileio import MerfishAnalysis, ImageDataset
from mftools.cellgene import create_scanpy_object
from mftools.segmentation import CellSegmentation
from mftools.barcodes import assign_to_cells, link_cell_ids, create_cell_by_gene_table

logging.basicConfig(
    filename="/storage/RNA_MERFISH/MERSCOPE/merscope.log",
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def prune_broken_links(folder) -> None:
    """Identify and remove links to experiments that have been removed from the NAS."""
    for path in folder.iterdir():
        logging.debug(f"Checking if {path} is broken link")
        if path.is_symlink() and not path.exists():
            path.unlink()
            logging.info(f"Removed broken link {str(path)}")


def link_subfolder(path, folder) -> None:
    subfolder = path / folder
    destfolder = pathlib.Path(f"/storage/RNA_MERFISH/MERSCOPE/{folder}/")
    if not subfolder.exists():
        return
    for exp in subfolder.iterdir():
        dest = destfolder / exp.stem
        if not dest.exists():
            dest.symlink_to(exp)
            logging.info(f"Link created {str(dest)} -> {str(exp)}")


def link_folder(path) -> None:
    if path.exists():
        link_subfolder(path, "Data")
        link_subfolder(path, "Analysis")
        link_subfolder(path, "Output")


def link_volume(nas_volume) -> None:
    link_folder(pathlib.Path(nas_volume, "RNA_MERFISH", "MERSCOPE"))
    link_folder(pathlib.Path(nas_volume, "MERSCOPE"))


def update_merscope_folders() -> None:
    logging.debug(f"Checking for broken links")
    #prune_broken_links(pathlib.Path("/storage/RNA_MERFISH/MERSCOPE/Data"))
    #prune_broken_links(pathlib.Path("/storage/RNA_MERFISH/MERSCOPE/Analysis"))
    #prune_broken_links(pathlib.Path("/storage/RNA_MERFISH/MERSCOPE/Output"))
    logging.debug(f"Checking for new data")
    for nas_volume in nas_volumes():
        link_volume(nas_volume)


def nas_volumes():
    for nas_volume in pathlib.Path("/mnt").iterdir():
        if str(nas_volume.stem).startswith("merfish") and not str(nas_volume.stem) in ["merfish2rna", "merfish2dna"]:
            logging.debug(f"Linking from {nas_volume}")
            yield nas_volume
        else:
            logging.debug(f"Skipping {nas_volume}")


def get_experiment_list():
    """Find all experiments with analysis results."""
    for nas_volume in nas_volumes():
        yield from pathlib.Path(nas_volume, "RNA_MERFISH", "MERSCOPE", "Output").glob("*")
        yield from pathlib.Path(nas_volume, "MERSCOPE", "Output").glob("*")


def check_merscope_analysis(exp: pathlib.Path) -> None:
    merscope_scanpy_path = exp / "merscope" / "raw_counts.h5ad"
    if not merscope_scanpy_path.exists():
        try:
            logging.info(f"Creating merscope scanpy object for {exp}")
            output = MerfishAnalysis(exp, save_to_subfolder="merscope")
            adata = create_scanpy_object(output)
            adata.write(merscope_scanpy_path)
        except Exception as e:
            logging.error(e)
    cellpose_scanpy_path = exp / "cellpose" / "raw_counts.h5ad"
    if not cellpose_scanpy_path.exists():
        datadir = pathlib.Path(f"/storage/RNA_MERFISH/MERSCOPE/Data/{exp.stem}")
        maskdir = pathlib.Path(exp / "masks")
        if maskdir.exists() or datadir.exists():
            try:
                logging.info(f"Creating cellpose scanpy object for {exp}")
                output = MerfishAnalysis(exp, save_to_subfolder="cellpose")
                if not output.has_cell_metadata():
                    channel = "PolyT"
                    zslice = 4
                    if datadir.exists():
                        imageset = ImageDataset(datadir)
                    else:
                        imageset = None
                    seg = CellSegmentation(maskdir, imagedata=imageset, channel=channel, zslice=zslice)
                    output.save_cell_metadata(seg.metadata)
                else:
                    seg = CellSegmentation(maskdir)
                barcodes = output.load_barcode_table()
                assign_to_cells(barcodes, seg)
                link_cell_ids(barcodes, seg.linked_cells)
                output.save_barcode_table(barcodes)
                cbgtab = create_cell_by_gene_table(barcodes)
                cbgtab.index = cbgtab.index.astype(int)
                output.save_cell_by_gene_table(cbgtab)
                adata = create_scanpy_object(output, positions=seg.positions)
                adata.write(cellpose_scanpy_path)
            except Exception as e:
                logging.error(e)


def check_all_analysis() -> None:
    for exp in pathlib.Path("/storage/RNA_MERFISH/MERSCOPE/Output/").iterdir():
        check_merscope_analysis(exp)


def create_website() -> None:
    html = """
    <!DOCTYPE html>
        <html>
            <body>

                <h2>Experiments</h2>
                <ul style="list-style-type:none;">
    """
    for exp in sorted(pathlib.Path("/storage/RNA_MERFISH/MERSCOPE/Output/").iterdir()):
        if pathlib.Path(exp, "experiment.json").exists():
            metadata = json.load(open(exp / "experiment.json"))
            name = f"{metadata['experimentName']} - {metadata['experimentDescription']}"
        else:
            name = exp.stem
        html += f"<li><a href='{exp.stem}'>{name}</a></li>\n"
        path = pathlib.Path("/var/www/html/merscope/", exp.stem)
        path.mkdir(exist_ok=True)
        html2 = "<!DOCTYPE html><html><body>"
        for i, png in enumerate(exp.glob("region_*/summary.png")):
            summary_path = pathlib.Path(f"/var/www/html/merscope/{exp.stem}/summary_{i}.png")
            if not summary_path.exists():
                pathlib.Path(f"/var/www/html/merscope/{exp.stem}/summary_{i}.png").symlink_to(png)
            html2 += f"<h2>Region {i}</h2><img src='summary_{i}.png'>"
        html2 += "</body></html>"
        with open(path / "index.html", "w") as f:
            print(html2, file=f)

    html += "</ul>\n</body>\n</html>"

    with open("/var/www/html/merscope/index.html", "w") as f:
        print(html, file=f)


def main() -> None:
    update_merscope_folders()
    check_all_analysis()
    create_website()


if __name__ == "__main__":
    main()
