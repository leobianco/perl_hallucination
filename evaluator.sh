python -m IPython -i evaluator.py \
  -- \
  --seed 12345 \
  --writer_model_base "google/gemma-2-2b-it" \
  --writer_model_lora "leobianco/halomi_writer_sft" \
  --max_tokens 256 \
  --evaluator_model "google/gemma-2-27b-it" \
  --num_fewshot_examples 4 \
  --evaluate_evaluator False \
  --threshold 0.144
