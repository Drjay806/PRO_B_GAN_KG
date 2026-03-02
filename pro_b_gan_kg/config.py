from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    delimiter: Optional[str] = None
    has_header: bool = False


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    compgcn_layers: int = 2
    dropout: float = 0.2
    comp_op: str = "mul"
    use_rgcn: bool = False
    fusion: str = "concat"
    attention_hidden: int = 256
    generator_hidden: int = 512
    discriminator_hidden: int = 512
    noise_dim: int = 64


@dataclass
class TrainingConfig:
    seed: int = 7
    batch_size: int = 1024
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs_pretrain: int = 5
    max_epochs_warmup: int = 10
    max_epochs_gan: int = 30
    patience_pretrain: int = 5
    patience_warmup: int = 10
    patience_gan: int = 15
    grad_clip: float = 1.0
    gan_k: int = 3
    lambda_adv: float = 0.1
    lambda_adv_max: float = 0.5
    lambda_adv_step: float = 0.05
    neighbor_dropout: float = 0.4
    leave_one_out: bool = True
    eval_topk: int = 1000


@dataclass
class SamplingConfig:
    easy_ratio: float = 0.5
    medium_ratio: float = 0.3
    hard_ratio: float = 0.2
    hard_update_interval: int = 5


@dataclass
class OptionalConfig:
    use_rl_evidence: bool = False
    use_patch_reranker: bool = False


@dataclass
class SemanticConfig:
    embeddings_dir: Optional[str] = None
    entity_type_mapping: Optional[Dict[str, str]] = None


@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optional: OptionalConfig = field(default_factory=OptionalConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RunConfig":
        data = DataConfig(**cfg["data"])
        model = ModelConfig(**cfg.get("model", {}))
        training = TrainingConfig(**cfg.get("training", {}))
        sampling = SamplingConfig(**cfg.get("sampling", {}))
        optional = OptionalConfig(**cfg.get("optional", {}))
        semantic = SemanticConfig(**cfg.get("semantic", {}))
        run = RunConfig(
            data=data,
            model=model,
            training=training,
            sampling=sampling,
            optional=optional,
            semantic=semantic,
        )
        run.validate()
        return run

    def validate(self) -> None:
        if self.model.comp_op not in {"mul", "sub", "add"}:
            raise ValueError("comp_op must be one of: mul, sub, add")
        if self.model.fusion not in {"concat", "gate"}:
            raise ValueError("fusion must be one of: concat, gate")
        if not (0.0 <= self.training.neighbor_dropout <= 1.0):
            raise ValueError("neighbor_dropout must be in [0, 1]")
        if abs(
            self.sampling.easy_ratio + self.sampling.medium_ratio + self.sampling.hard_ratio - 1.0
        ) > 1e-6:
            raise ValueError("Sampling ratios must sum to 1.0")
