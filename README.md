# spikesorting_pipeline

Pipeline for spike sorting using Kilosort4. Code attributions from Corbin Diaz, Diya Basrai, Wentao Qiu, Zahra Amer, and Peiyu Wang. Much of the pipeline is inspired and based on the Allen Institude of Neural Dynamics electrophysiology analysis pipeline with SpikeInterface.

## Setup

1. Clone the repo:

   ```
   git clone https://github.com/corbindiaz/spikesorting_pipeline.git
   cd spikesorting_pipeline
   ```

2. Setup your environment. You will need conda / miniconda installed:

   ```
   bash setup.sh
   conda env create -f environment.yml
   ```

   This will:
   - Create a local version of params.json from the default parameters. Then, `params.json` automatically picks up new parameters on future `git pull`s, without overwriting values you've already customized
   - Creates conda environment from `environment.yml`

3. You can then activate the environment:

   ```
   conda activate spikesort
   ```

## Running the Pipeline

This pipeline currently runs on SpikeGLX recordings. To run, specify where this recording lives in the command line, along with where you want results outputted:

```
python src/pipeline.py --recording "path/to/SpikeGLX/folder" --output "path/to/output/folder"
```

By default, the pipeline will run all steps: `preprocessing`, `kilosort`, `postprocess`, `export`, `curation`. However, you can also specify a to run only some steps, provided you have the necessary inputs. Preprocessing and Kilosort steps need a recording to run, while postprocess, export, and curation need an analyzer object. Specifically, Kilosort will look for a folder named "preprocessed" in your output folder, while the last three steps look for a folder named "analyzer".

```
python src/pipeline.py --output "path/to/output/folder" --steps postprocess
```

You can also specify to run everything after a certain step:

```
python src/pipeline.py --output "path/to/output/folder" --steps postprocess rest
```

Finally, export comes in two parts: `phy` and `bombcell`. You can run either individually as well, but `bombcell` requires the ouput of `phy` to run.

```
python src/pipeline.py --output "path/to/output/folder" --steps phy
python src/pipeline.py --output "path/to/output/folder" --steps bombcell
```

## Outline of Each Step

1. **Preprocessing** (`preprocess`)
   1. Trimming
   2. Cleaning
      1. Phase Shift
      2. Bandpass Filter
      3. Bad Channel removal
      4. CAR/CMR correction
   3. DREDge motion correction
2. **Kilsort4** (`kilosort`)
3. **Postprocessing** (`postprocess`)
   1. Remove excess channels
   2. Compute extensions
4. **Export** (`export`)
   1. Export contents for use of Phy (`phy`)
   2. Bombcell (`bombcell`)
      1. Compute Bombcell metrics for Phy
      2. Save unit plots
5. **Curation** (`curation`)
   1. (NOT FINISHED)

## Other

- `params.txt`: Gives brief comment explanations for unintuitive parameters
- `kilosort_to_analyzer.py`: Working code on converting kilosort output from a different pipeline into an analyzer object for postprocessing in this pipeline. This however is known to be giving bugs in terms of mapped amplitudes and channel positions.
- `remove_late_spikes.py`: Emergency code to remove extra spikes in phy-folders, assuming the extraneous spikes lie at the *end* of the recording. This should *never* be needed to be run unless you are using `kilosort_to_analyzer.py` or some adjacent code.