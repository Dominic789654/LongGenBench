#!/bin/bash

# Set default values
K=20
OUTPUT_PATH="./outputs/LongGenBench_MMLU/LongGenBench_MMLU_demo"
API_KEY=""
MODEL="gpt-4o-mini"
API_ENDPOINT=""

# Run the Python script with the specified arguments
python longgenbench_MMLU_openai.py \
    --k $K \
    --output_path "$OUTPUT_PATH" \
    --api_key "$API_KEY" \
    --model "$MODEL" \
    --api_endpoint "$API_ENDPOINT"