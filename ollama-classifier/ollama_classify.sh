#!/bin/bash
#SBATCH --job-name="ollama_classify"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output="logs/%x_o%j.%N.out"
#SBATCH --error="logs/%x_e%j.%N.err"
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

mkdir -p logs

# ---- Safety ----
set -euo pipefail

# ---- Require / default important env vars (set these outside) ----
: "${MODEL:=llama3}"                 # default if not exported
: "${INPUT_FILE:?Set INPUT_FILE}"    # hard-require INPUT_FILE
: "${SCRIPT:?Set SCRIPT}"            # hard-require SCRIPT (to classify)

set -x
echo "JobID=$SLURM_JOB_ID Host=$(hostname) Start=$(date)"
echo "MODEL=${MODEL-<unset>}"
echo "INPUT_FILE=${INPUT_FILE-<unset>}"
echo "SCRIPT=${SCRIPT-<unset>}"

# Load Modules
module purge
module load gpu
module load slurm

echo "Modules purged & gpu + slurm loaded"

# ---- Load Python/conda environment ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ollama

echo "Conda environment activated"

# ---- add ollama binary to path & start server ----
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="/expanse/lustre/scratch/$USER/temp_project/.ollama/models"
mkdir -p $OLLAMA_MODELS

echo "Updated Ollama's MODELS directory"

export OLLAMA_CONTEXT_LENGTH=16384
ollama serve > logs/ollama_serve.log &
OLLAMA_PID=$!
MAX_WAIT=30
WAITED=0

# Assuming port 11434
until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 0.5
    WAITED=$(echo "$WAITED + 0.5" | bc)
    if (( $(echo "$WAITED >= $MAX_WAIT" | bc -l) )); then
        echo "Ollama failed to start within $MAX_WAIT seconds"
        kill $OLLAMA_PID
        exit 1
    fi
done

echo "Ollama server is ready"

# check/download model
ollama list | grep -q "$MODEL" || ollama pull "$MODEL"

echo "Found/Downloaded model: $MODEL"

# Manage max threads
export OMP_NUM_THREADS=8

# ---- Define paths ----
SCRATCH_DIR="/scratch/$USER/job_$SLURM_JOB_ID"
LUSTRE_DIR="/expanse/lustre/scratch/$USER/temp_project/ollama/results"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$LUSTRE_DIR"

echo "Running inferencing script"

# ---- Run Python job ----
python3 -u "$SCRIPT" \
    --output_dir "$LUSTRE_DIR" \
    --input "$INPUT_FILE" \
    --model "$MODEL"

ollama ps

# ---- Copy results to Home ----
mkdir -p "$HOME/eeg_llm_classification"
cp "$LUSTRE_DIR/checkpoint_$MODEL.csv" "$HOME/eeg_llm_classification/checkpoint_$MODEL.csv"
cp "$LUSTRE_DIR/classified_$MODEL.xlsx" "$HOME/eeg_llm_classification/classified_$MODEL.xlsx"