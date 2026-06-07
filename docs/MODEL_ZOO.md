# BLUE Model Zoo

## SimLingo backbone

BLUE Stage 1 uses the official SimLingo backbone. The checkpoint is not committed
in this package. Download it from the official source and pass it with
`--agent-config` or set `SIMLINGO_CKPT`.

## BLUE SimLingo gate v1

- File: `gate/weights/blue_simlingo_gate.pt`
- Metadata: `gate/weights/blue_simlingo_gate.json`
- SHA256: `3d44af50d5f1639bc1e19fe3d7812d2f0fd6e9345893d3ccab83c420b9a4e788`
- Hidden size: `896`
- MLP hidden size: `128`
- Dropout: `0.5`
- Recommended threshold: `0.66`
- Public evaluation mode: `trained_gate`

The gate returns `1` for language generation and `0` for direct action.
