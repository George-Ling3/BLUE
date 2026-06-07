# BLUE Gate Weights

Stage 1 includes the default BLUE gate checkpoint for SimLingo:

- Checkpoint: `blue_simlingo_gate.pt`
- Metadata: `blue_simlingo_gate.json`
- Checksum file: `SHA256SUMS`
- Public mode: `trained_gate`
- Recommended threshold: `0.66`

Verify the checkpoint:

```bash
cd gate/weights
sha256sum -c SHA256SUMS
```

The gate input is the last language-token hidden state before language generation.
