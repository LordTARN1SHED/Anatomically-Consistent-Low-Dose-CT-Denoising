from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    mode: str = "synthetic"  # synthetic | dicom | npy_pairs | latent_cache
    image_size: int = 256
    batch_size: int = 2
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    clean_dir: Optional[str] = None
    noisy_dir: Optional[str] = None
    dicom_root: Optional[str] = None
    prepared_index_dir: Optional[str] = None
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    split_seed: int = 42
    min_slice_size: int = 64
    strict_ct_only: bool = True

    hu_clip_min: float = -1000.0
    hu_clip_max: float = 1000.0

    lung_window_width: float = 1500.0
    lung_window_level: float = -600.0
    soft_window_width: float = 350.0
    soft_window_level: float = 50.0

    noise_model: str = "signal_dependent_gaussian"
    noise_sigma_min: float = 0.01
    noise_sigma_max: float = 0.06
    noise_alpha: float = 0.02

    edge_blur_sigma: float = 1.0
    edge_low_threshold: int = 40
    edge_high_threshold: int = 120
    neighboring_slices: int = 0


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str] = None
    pretrained_vae_name_or_path: Optional[str] = None
    pretrained_unet_name_or_path: Optional[str] = None
    vae_subfolder: str = "vae"
    unet_subfolder: str = "unet"

    latent_scaling_factor: float = 0.18215
    latent_channels: int = 4
    sample_size: int = 32
    use_tiny_models: bool = True
    use_identity_autoencoder: bool = False
    condition_token_downsample: int = 1

    controlnet_conditioning_channels: int = 1
    enable_ldct_condition: bool = True

    vae_frozen: bool = True
    unet_frozen: bool = True


@dataclass
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    save_every_steps: int = 100
    log_every_steps: int = 20
    num_train_timesteps: int = 1000
    train_condition_projector: bool = False
    disable_edge_condition: bool = False
    aux_l1_weight: float = 0.0
    aux_gradient_weight: float = 0.0
    anchor_model_type: str = "none"  # none | redcnn
    anchor_checkpoint_path: Optional[str] = None
    anchor_condition_dropout: float = 0.0

    cache_latents: bool = False
    latent_cache_dir: str = "data/latent_cache"


@dataclass
class EvalConfig:
    num_eval_samples: int = 200
    save_predictions: bool = True
    compute_fid: bool = True
    dataset_split: str = "val"
    num_inference_steps: int = 50
    inference_mode: str = "img2img"  # img2img | from_noise | direct_x0
    strength: float = 0.35
    controlnet_conditioning_scale: float = 1.0
    direct_timestep: int = 50
    output_blend_alpha: float = 1.0
    edge_adaptive_blend_strength: float = 0.0
    edge_adaptive_blur_sigma: float = 0.0
    anchor_blend_alpha: float = 0.0


@dataclass
class ProjectConfig:
    seed: int = 42
    output_dir: str = "outputs/egldm"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


@dataclass
class EvalRunConfig:
    seed: int
    checkpoint_path: str
    train_config_path: str
    output_dir: str
    num_samples: int
    save_predictions: bool
    compute_fid: bool
    dataset_split: Optional[str] = None
    num_inference_steps: Optional[int] = None
    inference_mode: Optional[str] = None
    strength: Optional[float] = None
    controlnet_conditioning_scale: Optional[float] = None
    direct_timestep: Optional[int] = None
    output_blend_alpha: Optional[float] = None
    edge_adaptive_blend_strength: Optional[float] = None
    edge_adaptive_blur_sigma: Optional[float] = None
    anchor_blend_alpha: Optional[float] = None


def _dataclass_from_dict(cls: Any, payload: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in payload.items() if k in fields}
    return cls(**kwargs)


def load_project_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = _dataclass_from_dict(DataConfig, raw.get("data", {}))
    model = _dataclass_from_dict(ModelConfig, raw.get("model", {}))
    train = _dataclass_from_dict(TrainConfig, raw.get("train", {}))
    eval_cfg = _dataclass_from_dict(EvalConfig, raw.get("eval", {}))

    top = _dataclass_from_dict(ProjectConfig, raw)
    top.data = data
    top.model = model
    top.train = train
    top.eval = eval_cfg
    return top


def load_eval_run_config(path: str | Path) -> EvalRunConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dataclass_from_dict(EvalRunConfig, raw)
