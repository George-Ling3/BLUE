# Reproduction

Stage 1 reproduction is closed-loop evaluation with the released BLUE gate:

```bash
module load conda
conda activate simlingo
python scripts/verify_assets.py
bash gate/evaluation/eval_blue_full.sh \
  --route-range 0:1 \
  --agent-config /path/to/pytorch_model.pt \
  --carla-root /path/to/carla \
  --out-dir outputs/blue_eval_smoke
```

Training reproduction will be added in Stage 2 together with the released
training data and training code.
