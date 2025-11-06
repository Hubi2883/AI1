# AI1: Time-LLM for Time-Series Classification and Anomaly Detection

Status: research code, may require tweaks to run on your setup.

## Simple abstract

This project adapts Time-LLM to time-series classification and anomaly detection. Instead of only forecasting, we map time-series into an LLM’s latent space and add lightweight classification heads. We tested a functional prototype on custom datasets and prompts to see how different heads and hyperparameters affect results. Early trials suggest that using LLM embeddings for time-series is feasible and can be effective for both classification and forecasting.

Keywords: time-series, LLM, classification, anomaly detection, Time-LLM

Note: The original focus (forecasting) was shifted to classification/anomaly detection for the scientific study.

## What to run (latest main branch)

- Train/evaluate (supports classification/anomaly detection/forecasting):
  - [run_main.py](https://github.com/Hubi2883/AI1/blob/2f193634195bf36c7ba7f374fff41ca534a8c29f/run_main.py)
- Full-series forecasting inference and plotting:
  - [run_inference.py](https://github.com/Hubi2883/AI1/blob/2f193634195bf36c7ba7f374fff41ca534a8c29f/run_inference.py)
- Optional: M4 experiments and pretraining:
  - [run_m4.py](https://github.com/Hubi2883/AI1/blob/main/run_m4.py)
  - [run_pretrain.py](https://github.com/Hubi2883/AI1/blob/main/run_pretrain.py)
- DeepSpeed config:
  - [ds_config_zero2.json](https://github.com/Hubi2883/AI1/blob/main/ds_config_zero2.json)

## Quick start (keep it simple)

1) Prepare environment (minimal set)
- Python, PyTorch (CUDA if available), accelerate, deepspeed, numpy, pandas, matplotlib, tqdm.

2) Put your CSV data under ./dataset (e.g., ./dataset/ETTm1.csv).

3) Run a tiny training to verify things work (classification or anomaly_detection also supported):
```bash
python run_main.py \
  --task_name classification \
  --is_training 1 \
  --model_id test \
  --model_comment smoke \
  --model Autoformer \
  --data ETTm1 \
  --root_path ./dataset \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --batch_size 16 --eval_batch_size 8 \
  --train_epochs 1 --itr 1 \
  --prompt_domain 1 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --content "simple prompt"
```

- If this completes and prints validation/test stats, your setup is OK.
- For anomaly detection, set `--task_name anomaly_detection`.
- For forecasting, keep `--task_name long_term_forecast`.

4) (Optional) Forecasting-only inference and plot:
```bash
python run_inference.py \
  --model Autoformer \
  --checkpoint_path ./checkpoints/<your_saved_setting>-smoke/ckpt.pth \
  --data ETTm1 \
  --data_path ./dataset/ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --d_model 16 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 32 \
  --factor 1 --embed timeF --freq h \
  --num_workers 4 --percent 100 --csv_name ETTm1 \
  --llm_dim 4096 --llm_layers 6 --llm_model LLAMA \
  --prompt_domain 1 --content "simple prompt" \
  --patch_len 16 --stride 8 --dropout 0.1 \
  --task_name long_term_forecast --batch_size 8 \
  --output_attention --device cuda \
  --save_plot_path forecasting_results.png \
  --seasonal_patterns Monthly
```

Tip: use small epochs/batches first. If DeepSpeed/Accelerate causes issues, reduce batch sizes or adjust the config in `ds_config_zero2.json`.

## Notes

- Code is research-grade; some paths/args may need adjustment for your data.
- Classification and anomaly detection were the study’s main focus; forecasting scripts are kept for reference.
- Prompts are loaded via `utils.tools.load_content`; keep `--content` minimal when testing.

If anything fails, try:
- Smaller `--train_epochs`, `--batch_size`.
- Verify `--features`, `--target`, and CSV columns match your data.
- Run on CPU first by setting `--device cpu` (slower, but simpler).
