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
    max_steps_per_epoch: int = -1  # -1 = no limit; set >0 to cap batches for test runs
    compgcn_edge_sample_ratio: float = 1.0  # fraction of edges passed to CompGCN each epoch; 1.0 = all edges
    encoder_refresh_interval: int = 5  # update CompGCN weights every N epochs; 0 = frozen
    encoder_refresh_sample_ratio: float = 0.20  # fraction of edges used during the encoder gradient update
    max_eval_samples: int = -1  # -1 = all val triples; set >0 to cap for faster test evaluation
    resume: bool = True  # if True, auto-resume from last saved checkpoint if output_dir already has one
    max_neighbors: int = 64  # max neighbor context per entity in attention; higher = more context, more memory


@dataclass
class SamplingConfig:
    easy_ratio: float = 0.5
    medium_ratio: float = 0.3
    hard_ratio: float = 0.2
    hard_update_interval: int = 5


@dataclass
class RLConfig:
    enabled: bool = False
    max_epochs: int = 10
    patience: int = 5
    lr: float = 1e-4
    budget: int = 3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    baseline_decay: float = 0.95
    batch_size: int = 256
    hub_penalty: float = 0.1
    proximity_bonus: float = 0.1
    max_triples_per_epoch: int = 10000
    policy_hidden: int = 0


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
    rl: RLConfig = field(default_factory=RLConfig)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RunConfig":
        data = DataConfig(**cfg["data"])
        model = ModelConfig(**cfg.get("model", {}))
        training = TrainingConfig(**cfg.get("training", {}))
        sampling = SamplingConfig(**cfg.get("sampling", {}))
        optional = OptionalConfig(**cfg.get("optional", {}))
        semantic = SemanticConfig(**cfg.get("semantic", {}))
        rl = RLConfig(**cfg.get("rl", {}))
        run = RunConfig(
            data=data,
            model=model,
            training=training,
            sampling=sampling,
            optional=optional,
            semantic=semantic,
            rl=rl,
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
