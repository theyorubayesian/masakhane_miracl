export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="masakhane-miracl"
export WANDB_ENTITY="masakhane-miracl"
export WANDB_API_KEY="<your-api-key>"

python -m tevatron.driver.train \
--output_dir models/swahili-bert-msmarco/ \
--model_name_or_path Davlan/bert-base-multilingual-cased-finetuned-swahili \
--save_steps 20000 \
--dataset_name Tevatron/msmarco-passage \
--fp16 \
--per_device_train_batch_size 8 \
--train_n_passages 8 \
--learning_rate 5e-6 \
--q_max_len 16 \
--p_max_len 128 \
--num_train_epochs 10 \
--logging_steps 500 \
--overwrite_output_dir \
--report_to wandb
