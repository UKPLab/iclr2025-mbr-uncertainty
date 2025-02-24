from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default=None
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    method: str = field(default=384)

    task: str = field(default=None)
    prompt_prefix: str = field(default="")

    # Seq2Seq model specific args
    generation_max_len: int = field(default=128)
    generation_beam_size: int = field(default=4)
    generation_do_sample: bool = field(default=False)
    generation_length_penalty: float = field(default=1.0)
    generation_top_k: int = field(default=50)
    generation_no_repeat_ngram_size: int = field(default=3)
    num_return_sequences: int = field(default=1)
    num_sequences_to_keep: int = field(default=1)
    num_labels: int = field(default=None)

    # Tokenization
    max_input_length: int = field(default=1024)
    max_output_length: int = field(default=1024)

    dropout: float = field(default=0.1)
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None,
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_filter_dict: Optional[dict] = field(
        default=None,
    )
    dataset_train_split: str = field(default="train")
    dataset_val_split: Optional[str] = field(default="validation")
    dataset_test_split: Optional[str] = field(default="validation")

    is_training: bool = field(default=True)

    ivon_s: Optional[str] = field(default=None)

    lr_scheduler: Optional[str] = field(default="cosine")


@dataclass
class DataPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_filter_dict: Optional[dict] = field(
        default=None,
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None,
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_test_split: str = field(default="test")

    test_documents_faiss_index_path: str = field(default=None)

    metric_output_file: Optional[str] = field(default=None)

    prediction_output_file: Optional[str] = field(default=None)

    is_training: bool = field(default=False)

    ivon_s: Optional[str] = field(default=None)

@dataclass
class OptimizerArguments:
    optimizer_name: Optional[str] = field(default="AdamW")

    beta1: Optional[float] = field(default=0.9)
    beta2: Optional[float] = field(default=0.95)
    eps: Optional[float] = field(default=1e-6)

    grad_clip: Optional[float] = field(default=10000.0)
    rho: Optional[float] = field(default=10000.0)
    mc_samples: Optional[int] = field(default=1)
    inference_mc_samples: Optional[int] = field(default=1)
    hess_init: Optional[float] = field(default=1.0)
    damping: Optional[float] = field(default=0.0)
    temperature: Optional[float] = field(default=0.1)
    normalized_weightdecay: Optional[float] = field(default=25.0)
    anneal_steps: Optional[int] = field(default=1)
    clip_radius: Optional[float] = field(default=10000.0)
    min_lr: Optional[float] = field(default=0.0)
    prec_lr_scale: Optional[float] = field(default=None)
    data_size_scaling_factor: Optional[float] = field(default=1.0)
    lr2: Optional[float] = field(default=0.001)
    std_max: Optional[float] = field(default=100000.0)
    ess: Optional[float] = field(default=1000000.0)
    use_prior_mean: Optional[bool] = field(default=False)
    std_scale: Optional[float] = field(default=None)
    kl_scale: Optional[float] = field(default=None)
    hessian_estimator: Optional[str] = field(default="price")
    rescale_lr: bool = field(default=False)
    use_peft: bool = field(default=False)
    load_peft_model: bool = field(default=False)