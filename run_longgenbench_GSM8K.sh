#!/bin/bash

# # Set default values
K=35
PROMPT_PATH="./data/LongGenBench_GSM8K_prompt/LongGenBench_prompt_json.txt"
OUTPUT_PATH="./outputs/LongGenBench_GSM8K/LongGenBench_GSM8K_demo.txt"
QUESTION_LIMIT=700
API_KEY=""
MODEL="gpt-4o-mini"
API_ENDPOINT=""

# Run the Python script with the specified arguments
python longgenbench_GSM8K_openai.py \
    --k $K \
    --prompt_path "$PROMPT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --question_limit $QUESTION_LIMIT \
    --api_key "$API_KEY" \
    --model "$MODEL" \
    --api_endpoint "$API_ENDPOINT"



