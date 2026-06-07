"""Hydra config registrations for the public BLUE runtime."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class VLMEncoderConfig:
    variant: str = "OpenGVLab/InternVL2-1B"
    embed_dim: int = 512
    freeze: bool = False
    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = "OpenGVLab/InternVL2-1B"
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1
    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any
    lr: float = 5e-2
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = "2d"
    predict_route_as_wps: bool = True
    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DrivingModelGateConfig:
    """Config for the BLUE gated SimLingo model."""

    vision_model: Any = None
    language_model: Any = None
    lr: float = 5e-2
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = "2d"
    predict_route_as_wps: bool = True
    gate_mode: str = "trained_gate"
    gate_ckpt: Optional[str] = None
    gate_kwargs: Optional[Dict[str, Any]] = None
    _target_: str = "simlingo_training.models.driving_gate.DrivingModelGate"


@dataclass
class DatasetBaseConfig:
    data_path: str = "database/simlingo_v2_*"
    bucket_path: str = "data/buckets"
    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False
    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False
    skip_first_n_frames: int = 10
    pred_len: int = 11
    hist_len: int = 1
    hist_len_commentary: int = 5
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    use_safety_flag: bool = False
    num_route_points: int = 20
    route_as: str = "target_point_command"
    use_lmdrive_commands: bool = True


@dataclass
class DrivingDataModuleConfig:
    base_dataset: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    batch_size: int = 16
    num_workers: int = 10
    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model", name="driving_gate", node=DrivingModelGateConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
