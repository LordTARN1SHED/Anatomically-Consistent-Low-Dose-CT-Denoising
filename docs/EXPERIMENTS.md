# Key Experiment Commands

Run all commands from the root of this release folder.

## 1. Prepare real LIDC-IDRI metadata

```bash
python code/scripts/prepare_lidc.py --config code/configs/train_lidc_hybrid_medium.yaml
```

## 2. Sanity checks

```bash
python code/scripts/verify_zero_conv.py --config code/configs/train_lidc_hybrid_medium.yaml
python code/scripts/check_vae_reconstruction.py --config code/configs/train_lidc_hybrid_medium.yaml --batches 4
```

## 3. Train the RED-CNN anchor baseline

```bash
python code/scripts/train_redcnn.py --config code/configs/train_redcnn_medium.yaml
python code/scripts/evaluate_redcnn.py --config code/configs/eval_redcnn_medium_test.yaml
```

## 4. Train the final Hybrid EG-LDM

```bash
python code/scripts/train_controlnet.py --config code/configs/train_lidc_hybrid_medium.yaml
python code/scripts/evaluate.py --config code/configs/eval_lidc_hybrid_medium_val_best.yaml
python code/scripts/evaluate.py --config code/configs/eval_lidc_hybrid_medium_test_best.yaml
```

## 5. Run the original diffusion-family baselines

```bash
python code/scripts/train_controlnet.py --config code/configs/train_lidc_medium.yaml
python code/scripts/train_controlnet.py --config code/configs/train_lidc_medium_no_edge.yaml
python code/scripts/evaluate.py --config code/configs/eval_lidc_medium_test_tuned.yaml
python code/scripts/evaluate.py --config code/configs/eval_lidc_medium_no_edge_test.yaml
```

## 6. Re-run lightweight inference tuning

```bash
python code/scripts/tune_inference.py \
  --train_config code/configs/train_lidc_hybrid_medium.yaml \
  --checkpoint outputs/egldm_lidc_hybrid_medium/checkpoints/controlnet_best.pt \
  --out_dir outputs/hybrid_anchor_local_sweep_val128 \
  --num_samples 128 \
  --steps 1 \
  --strengths 0.0 \
  --scales 0.8 \
  --mode direct_x0 \
  --direct_timesteps 20 \
  --blend_alphas 1.0 \
  --anchor_blend_alphas 0.03,0.05,0.07 \
  --edge_strengths 0.05,0.1,0.15 \
  --edge_blur_sigmas 0.0 \
  --objective balanced \
  --dataset_split val
```
