python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file /lamport/makkapakka/jingxi_chen/GPT2/NLG/trained_models/GPT2_M/e2e/predict.26289.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt