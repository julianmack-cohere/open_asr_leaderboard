#!/bin/bash
set -e
export PYTHONPATH="..":$PYTHONPATH
RUNDIR=`pwd`
MODEL_IDs=(
    "CohereLabs/cohere-transcribe-03-2026"
)
BATCH_SIZE=64  # must be lower due to this issue: https://github.com/pytorch/pytorch/issues/80020
# We need something like _conv_split_by_batch here to fix this:
# https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/blob/main/modeling_cohere_asr.py#L138
DEVICE_ID=0

num_models=${#MODEL_IDs[@]}

echo "========================================================"
echo "Starting Cohere ASR English Evaluation"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE_ID"
echo ""


for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    if [[ -d "$MODEL_ID" ]]; then # local path
        short_model_id="${MODEL_ID##*/}"
    else # HF repo id
        short_model_id="${MODEL_ID#*/}"
    fi
    RESULTS_DIR="${RUNDIR}/benchmarks/results-${short_model_id}/en"
    
    # EDIT: the dataset to ami|voxpopuli 
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=1024 \
        --basedir="$RESULTS_DIR"
    exit 0
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_ID} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --basedir="$RESULTS_DIR"

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"
    
    # Evaluate results 
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RESULTS_DIR}', '${MODEL_ID}')" && \
    cd $RUNDIR

done


echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
