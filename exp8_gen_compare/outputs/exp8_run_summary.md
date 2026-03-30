# Exp8 Multi-GPU Run Summary

- Date: 2026-03-29
- Environment: conda `machine`
- Mode: distributed training (DDP), 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`)
- Command: `torchrun --standalone --max_restarts=0 --nproc_per_node=4 exp8_gen_compare/src/train_compare_models.py --profile gpu_hq_12g --quick-test`

## Outcome

- Run status: completed
- Profile used: `gpu_hq_12g`
- Quick-test: `true`
- World size: `4`

## Metrics (from comparison_metrics.csv)

- diffusion: FID=437.7056, train_time_min=0.100, params_m=7.330
- flow_matching: FID=501.1480, train_time_min=0.068, params_m=7.330
- autoregressive: FID=421.5304, train_time_min=0.063, params_m=2.109

## Ranking (lower FID is better)

1. autoregressive (421.5304)
2. diffusion (437.7056)
3. flow_matching (501.1480)

## Key artifacts

- `exp8_gen_compare/outputs/exp8_multigpu_run.log`
- `exp8_gen_compare/outputs/comparison_metrics.csv`
- `exp8_gen_compare/outputs/ranking.csv`
- `exp8_gen_compare/outputs/run_summary.json`
- `exp8_gen_compare/report8.tex`
- `exp8_gen_compare/figures/fid_barplot.png`
- `exp8_gen_compare/figures/ultra_curated_comparison.png`
- `exp8_gen_compare/figures/class_panels_cat_dog_car_hq.png`
