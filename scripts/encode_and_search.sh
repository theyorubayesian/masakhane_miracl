export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_CACHE=

# This script only runs the `Search` only if it successfully completed both the `Query & Passage Encoding``
# If any or both encodings have already completed, uncomment either or both or lines 5 & 6, & the relevant command(s)
# enc_query=true
# enc_corpus=true

# This script only runs evaluation if search has been completed.
# Uncomment the line below and the commands for encoding and search to evaluate only.
# faiss_search=true
PROJECT_PATH="."
dataset_name="miracl"
hf_query_dataset_name="miracl/miracl"
hf_corpus_dataset_name="miracl/miracl-corpus"
model_name="afriberta-base-msmarco-miracl-sw"
splits=("dev" "testA" "testB")
language="swahili"
hf_language_code="sw"
eval_qrels="mrtydi-v1.0-$language/qrels.test.txt"
models=()

# set -n; # No execution. Error checking alone. Uncomment to run.
# set -x; # Print each line before running it

OUTPUT_PATH="$PROJECT_PATH/outputs/$model_name/"
RUNS_PATH="runs/$model_name/$language/"
mkdir -p {"$OUTPUT_PATH/corpus_encoding/$language","$OUTPUT_PATH/query_encoding/$language",$RUNS_PATH}

for split in $splits
do
  query_embd_path="$PROJECT_PATH/outputs/$model_name/query_encoding/$language/${dataset_name}_${split}_query_emb.pkl"
  corpus_embd_path="$PROJECT_PATH/outputs/$model_name/corpus_encoding/$language/${dataset_name}_corpus_emb.pkl"

  # Query Encoding
  python -m tevatron.driver.encode \
  --model_name_or_path $PROJECT_PATH/models/$model_name/ \
  --output_dir outputs/$model_name/query_encoding \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --dataset_name "$hf_query_dataset_name:$hf_language_code/$split" \
  --encoded_save_path $query_embd_path \
  --q_max_len 64 \
  --encode_is_qry && enc_query=true

  # Corpus Encoding
  if [[ ! -f corpus_embd_path ]]; then
    python -m tevatron.driver.encode \
    --output_dir $PROJECT_PATH/outputs/$model_name/corpus_encoding \
    --model_name_or_path $PROJECT_PATH/models/$model_name/ \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --p_max_len 256 \
    --dataset_name $hf_corpus_dataset_name:$hf_language_code \
    --encoded_save_path $corpus_embd_path \
    --encode_num_shard 1 && enc_corpus=true
  else
    enc_corpus=true
  fi

  # Search
  if [[ $enc_corpus && $enc_query ]]; then
      python -m tevatron.faiss_retriever \
      --query_reps $query_embd_path \
      --passage_reps $corpus_embd_path \
      --depth 100 \
      --batch_size -1 \
      --save_text \
      --save_ranking_to "runs/$model_name/$language/$dataset_name.$split.txt" && \
      python -m tevatron.utils.format.convert_result_to_trec \
      --input "runs/$model_name/$language/$dataset_name.$split.txt" \
      --output "runs/$model_name/$language/$dataset_name.$split.trec" && \
      faiss_search=true
  fi
done

# Evaluation
if [[ $faiss_search ]]; then
  python -m pyserini.eval.trec_eval -c \
  -mrecip_rank \
  -mrecall.100 \
  $eval_qrels \
  "runs/$model_name/$language/$dataset_name.$split.trec" > "runs/$model_name/$language/$dataset_name.$split.results"