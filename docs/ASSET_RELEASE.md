# BLUE Asset Release Notes

## Current public release

The current public release corresponds to Stage 1 of the open-source plan. The
repository includes the assets needed for closed-loop evaluation:

- `gate/weights/blue_simlingo_gate.pt`
- `gate/weights/blue_simlingo_gate.json`
- `gate/weights/SHA256SUMS`
- `data/routes/bench2drive_split/`
- `evaluation_logs/bench2drive/`

## External assets

The official SimLingo backbone checkpoint should be downloaded from its official
source and must keep the original license and attribution.

## Upcoming release scope

Stage 2 will add the BLUE training data and training code. The repository no
longer advertises additional placeholder release stages beyond that scope.

## Checksum generation

```bash
sha256sum gate/weights/blue_simlingo_gate.pt > gate/weights/SHA256SUMS
python scripts/verify_assets.py
```

Update `configs/assets.yaml` whenever an asset URL or checksum changes.
