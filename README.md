# BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving

[![Project Page](https://img.shields.io/badge/Project%20Page-Blue-0A66C2?logo=googlechrome&logoColor=white)](https://blue-website.github.io/)
[![paper](https://img.shields.io/badge/arXiv-2606.08684-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.08684)
[![Hugging Face Weights](https://img.shields.io/badge/Hugging%20Face-Weights-yellow?logo=huggingface)](https://huggingface.co/George-Ling/blue_gate)
[![Hugging Face Data](https://img.shields.io/badge/Hugging%20Face-Data-yellow?logo=huggingface)](https://huggingface.co/datasets/George-Ling/blue_data)
[![Hugging Face Logs](https://img.shields.io/badge/Hugging%20Face-Logs-yellow?logo=huggingface)](https://huggingface.co/George-Ling/blue_gate/tree/main/evaluation_logs)

This repository is the official codebase for our EMNLP paper "BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving".

TLDR: Driving VLAs often generate language reasoning that is useless or even harmful to driving. 
BLUE addresses this by generating language only when it clearly helps, thereby improving driving performance while reducing inference latency.

BLUE uses a 0.11M-parameter gate to decide at each frame whether to predict driving actions with or without intermediate language generation.

## 🎉 News

2026-08 - Our paper was accepted to the [EMNLP 2026 Main Conference](https://2026.emnlp.org/program/main_papers/). 🎉

2026-06 - We released the [Project Page](https://blue-website.github.io). It includes some demo videos. 🎉Please check it!

2026-06 - We released the BLUE [evaluation code](gate/evaluation/eval_blue_full.sh), [model checkpoints](gate/weights/blue_simlingo_gate.pt), and [evaluation logs](evaluation_logs/bench2drive/bench2drive_merged_2.json). 🎉Please try it!

## ⚙️ Environment Setup

Create a Python environment and install the packages listed in `requirements.txt`.

```bash
module load conda
conda create -n blue python=3.8 -y
conda activate blue
python -m pip install -r requirements.txt
```

Install CARLA 0.9.15 from the official CARLA release page:
[[CARLA 0.9.15]](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

After installation, set the CARLA root to the directory that contains:

```text
CarlaUE4.sh
PythonAPI/carla/
```

## 📦 Weight Download

The BLUE gate checkpoint is already included in this repository at
[`gate/weights/blue_simlingo_gate.pt`](gate/weights/blue_simlingo_gate.pt).

To use the SimLingo backbone, download the official checkpoint from the
[official SimLingo repository](https://huggingface.co/RenzKa/simlingo/tree/main/simlingo/checkpoints/epoch%3D013.ckpt), then pass the local
`pytorch_model.pt` path through `--agent-config` when running evaluation.

You can verify bundled assets with:

```bash
python scripts/verify_assets.py
```

Model checkpoints and evaluation logs will also be mirrored on Hugging Face:
[[Weights]](https://huggingface.co/George-Ling/blue_gate) | [[Data]](https://huggingface.co/datasets/George-Ling/blue_data)

## 🚀 Quick Start

#### Static checks

```bash
cd blue
module load conda
conda activate blue

bash -n gate/evaluation/eval_blue_full.sh
python scripts/verify_assets.py
python tests/smoke/test_result_summary.py
python tests/smoke/test_gate_checkpoint.py
```

#### One-route closed-loop smoke test

```bash
cd blue
module load conda
conda activate blue

bash gate/evaluation/eval_blue_full.sh \
  --route-range 0:1 \
  --agent-config /path/to/pytorch_model.pt \
  --carla-root /path/to/carla \
  --out-dir outputs/blue_eval_smoke
```

## 📁 Repository Map

```text
blue/
├── data/
│   ├── README.md                         # data release status and layout
│   └── routes/bench2drive_split/         # 220 Bench2Drive route XMLs
├── gate/
│   ├── evaluation/eval_blue_full.sh      # closed-loop evaluation entry point
│   ├── runtime/                          # decision-log utilities
│   └── weights/                          # BLUE gate checkpoint
├── simlingo_training/models/
│   ├── gate.py                           # BLUE gate runtime
│   └── driving_gate.py                   # SimLingo gate integration
├── team_code/agent_simlingo.py           # Bench2Drive agent
├── Bench2Drive/                          # evaluator components
├── evaluation_logs/                      # released evaluation logs
├── configs/                              # asset and evaluation configs
├── docs/                                 # auxiliary notes
├── tests/smoke/                          # smoke tests
└── requirements.txt                      # package snapshot
```

## 📊 Framework and Results

### Framework

![BLUE framework](figure/framework.png)

### Results on Bench2Drive

![BLUE main result on Bench2Drive](figure/exp_result1.png)

### Results on Longest & Latency Comparison

![BLUE longest result and inference efficiency/latency comparison](figure/exp_result2.png)

### 📋 Ready-to-Cite Results

Bench2Drive and Fail2Drive results are reported as mean ± std over three seeds; Longest6 v2 and NAVSIM use a single seed.

#### BLUE on Bench2Drive

| SR (%) ↑ | DS ↑ | Efficiency (%) ↑ | Smoothness ↑ |
|---:|---:|---:|---:|
| 76.18 ± 0.64 | 90.58 ± 0.12 | 256.63 ± 2.48 | 0.2524 ± 0.0162 |

| Merge ↑ | Overtake ↑ | EmBrake ↑ | GiveWay ↑ | TSign ↑ | Mean ↑ |
|---:|---:|---:|---:|---:|---:|
| 61.44 ± 1.33 | 80.00 ± 1.81 | 93.27 ± 1.33 | 50.00 ± 0.00 | 84.74 ± 0.00 | 73.89 ± 0.14 |


#### BLUE on Longest6 v2

| Driving Score ↑ | Route Completion ↑ | Infraction Score ↑ |
|---:|---:|---:|
| 36.0 | 84.0 | 0.43 |

#### BLUE on Fail2Drive

| Split | DS ↑ | SR (%) ↑ | HM ↑ |
|---|---:|---:|---:|
| In Distribution | 85.67 ± 1.46 | 84.00 ± 3.27 | 84.81 ± 2.35 |
| Generalization | 73.87 ± 0.31 | 59.00 ± 1.63 | 65.59 ± 1.11 |

#### BLUE on NAVSIM

| EP ↑ | NC ↑ | DAC ↑ | DDC ↑ | TTC ↑ | Comfort ↑ | PDMS ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 81.35 | 98.50 | 94.77 | 97.88 | 94.84 | 99.99 | 87.00 |


## 📚 Citation

If you find BLUE useful, please consider citing our work:

```bibtex
@article{ling2026blue,
  title={BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving},
  author={Ling, George and Yang, Lijin and Yang, Hao and Huang, Zhongzhan},
  journal={arXiv preprint arXiv:2606.08684},
  year={2026}
}
```
