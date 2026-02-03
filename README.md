# eeg_llm_classification
tools for data de-identification, running the llm-classifier on SDSC, data-labeling &amp; manual validation, and performing clinical statistics.

## deidentify/
<span style="font-size: 18px;"> **deidentify.py**  <br></span>
To run, build and create the conda environment from deidentify.yaml.

```bash
conda env create -f deidentify.yaml
conda activate deidentify
```

Then, execute the python script followed by the spreadsheet containing the clinical notes and number of desired batches to run in parallel

```bash
python3 deidentify.py -f path/to/spreadsheet.xlsx -n 4
```

If unspecified, the script will execute deidentification as 4 parallel batch jobs.


## ollama-classifier/
<span style="font-size: 18px;"> **ollama_classification.ipynb**  <br></span>
After logging on to the Expanse SDSC, first make sure you have the miniconda distribution installed.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
You can opt to have conda initialized automatically, or add it to your .bashrc file later. Restart the terminal,

```bash
source ~/.bashrc
```

Install mamba for quicker environment-building. Then, create the ollama conda environment from the .yaml file:

```bash
conda install -n base -c conda-forge mamba -y
mamba env create -f ollama.yaml
conda activate ollama
```

Next, download the ollama binary for linux. On the cluster, tar does not have --zstd to decompress and read tar.zst files. However, miniconda3 provides you with the unzstd CLI tool:

```bash
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
  | unzstd \
  | tar x -C $HOME/$USER/.local
```

The batch job specifies ```OLLAMA_MODELS,``` which is where your models will be stored. However, you can manually configure it to lustre, where there is more storage by putting this in ```~/.bash_profile```:

```bash
export OLLAMA_HOME="export OLLAMA_MODELS="/expanse/lustre/scratch/$USER/temp_project/.ollama/models"
```

Before running the batch script, be sure to export the ```$INPUT_FILE```, ollama ```$MODEL```, and ```$SCRIPT``` that you are using to classify. If ```$MODEL``` is unspecified, llama3 will run. ```$INPUT_FILE``` and ```$SCRIPT``` are required.
```bash
export MODEL="llama3"
export INPUT_FILE="/$HOME/$USER/eeg_llm_classification/EEG_PHI_REMOVED.xlsx"
export SCRIPT="/home/$USER/eeg_llm_classification/scripts/ollama_classify.py"
```

And launch the job with sbatch. Account name can be found with ```sacctmgr show assoc user=$USER```
```bash
sbatch --account=abc123 ollama_classify.sh
```

## validation/
<span style="font-size: 18px;"> **label.py**  <br></span>

Run the script with:
```bash
python3 label.py -f path/to/deidentified_spreadsheet.xlsx
```
Not specifying spreadsheet location will automatically check the path ```../deidentify/EEG_PHI_removed.xlsx```