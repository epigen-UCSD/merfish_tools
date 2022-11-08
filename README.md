# merfish_tools -- under development
Functions and scripts for processing RNA MERFISH data.

## Running the pipeline

The script run.py is a pipeline that takes the molecules identified by MERlin and a cellpose segmentation and produces a number of quality metrics and outputs, notably a cell by gene table that can be used for further analysis. A scanpy object is also created with an initial clustering using default parameters.

To run the pipeline, a configuration file is required. See config.json for an example of this configuration.

This command will run the pipeline:

`python run.py -c config.json -o output_folder -e experiment_name`

The output of the pipeline will be put into the specified output folder. The input to the pipeline is defined by the analysis_root, data_root, segmentation_root, and segmentation_name specific in the configuration file, along with the experiment name given in the command. For example, with an analysis_root set to `/home/myself/merlin`, a data_root set to `/home/myself/data`, a segmentation_root set to `home/myself/segmentation`, a segmentation_name set to `cellpose_segmentation` and an experiment name of `exp1`, the pipeline will expect the find:
* Raw image files in `/home/myself/data/exp1`
* MERlin output in `/home/myself/merlin/exp1`
* Segmentation masks in `/home/myself/segmentation/exp1/cellpose_segmentation`

Documentation is a work in progress and will be expanded over time. Contact Colin Kern (jckern@ucsd.edu) for any questions.
