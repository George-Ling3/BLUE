# Troubleshooting

## `--agent-config` is missing

Pass the official SimLingo `pytorch_model.pt` checkpoint:

```bash
bash gate/evaluation/eval_blue_full.sh --agent-config /path/to/pytorch_model.pt --carla-root /path/to/carla
```

## CARLA import fails

Check `--carla-root` and make sure the directory contains `PythonAPI/carla`.
The evaluation script adds the CARLA Python paths to `PYTHONPATH`.

## Gate checkpoint checksum fails

Run:

```bash
cd gate/weights
sha256sum -c SHA256SUMS
```

If it fails, replace `blue_simlingo_gate.pt` with the released checkpoint.

## A route hangs or fails

Closed-loop CARLA evaluation can be unstable. Rerun the same route range or
increase `--max-retry`. Use smaller route slices such as `--route-range 0:1` for
smoke tests.
