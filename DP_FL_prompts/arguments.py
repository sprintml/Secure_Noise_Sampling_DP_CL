import copy

import dp_transformers
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import List
from typing import Optional
from tasks.utils import *
import numpy as np


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    label_name: str = field(
        default='labels',
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    label_names: Optional[List[str]] = field(
        default=('labels',),
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    remove_unused_columns = False

    training_type: str = field(
        default='public',
        # default='private',
        metadata={
            "help": "The type of training. public or private"
        }
    )

    privacy_engine: str = field(
        # default='dp_transformers',
        default='private_transformers',
        metadata={
            "help": "The type of the privacy engine to use."
        }
    )
    lr_decay: str = field(default="no", metadata={"help": "Apply the usual linear decay if `yes`, otherwise no deacy."})
    evaluate_before_training: bool = field(default=False, metadata={"help": "Run evaluation before training."})
    evaluate_after_training: bool = field(default=False, metadata={"help": "Run evaluation after training."})
    eval_epochs: int = field(default=10, metadata={"help": "Evaluate once such epochs"})
    evaluate_test_split: bool = field(default=False, metadata={"help": "Run evaluation on the test split"})
    slurm_file: str = field(default='unspecified', metadata={"help": "The name of the slurm file to write the log."})


@dataclass
class CollaborativeLearningArguments:
    # num_global_server_steps: int = field(
    #     default=10,
    #     metadata={
    #         "help": "number of epochs the aggregate model get updated."
    #     }
    # )
    num_global_server_epochs: int = field(
        default=21,
        metadata={
            "help": "number of epochs the aggregate model get updated."
        }
    )


    num_clients: int = field(
        default=10,
        metadata={
            "help": "number of clients."
        }
    )

    client_data_split_ratio: str = field(
        default="1/10",
        metadata={
            "help": "list of split ratios seperated by commas. E.g., '1/10', '1/2, 1/6, 1/6, 1/6' "
        }
    )

    how_client_sample_data: str = field(
        default="same_sampling_rate",
        metadata={
            "help": "number of clients."
        }
    )

    client_local_train_epoch_or_step: int = field(
        default=1,
        metadata={
            "help": "whether clients train for n epochs or n steps. 1 for steps and 0 for epochs. Default is 1 for steps"
        }
    )

    num_local_client_epoch_or_step: int = field(
        default=1,
        metadata={
            "help": "number of epochs/steps the client model gets updated each"
        }
    )

    sample_clients_ratio: float = field(
        default=1.0,
        metadata={
            "help": "the fraction of clients to update their updates during each aggregation round"
        }
    )

    save_freq_aggregate: int = field(
        default=10,
        metadata={
            "help": "number of steps the client model gets updated each"
        }
    )

    eval_freq_aggregate: int = field(
        default=1,
        metadata={
            "help": "Evaluate once such epochs"
        }
    )

    private_type: str = field(
        default='bound_datapoint',
        # default='bound_client',
        metadata={
            "help": "granularity of privacy"
        }
    )

    global_output_dir: str = field(
        default='checkpoints',
        metadata={
            "help": "where to save everything"
        }
    )

    global_overwrite_output_dir: int = field(
        default=0,
        metadata={
            "help": "override global output directory"
        }
    )

    global_resume_from_checkpoint: str = field(
        default=0,
        metadata={
            "help": "global resume from checkpoint"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    template: Optional[str] = field(
        default=None, metadata={"help": "template of the prompts for prompt-infill or prefix-infill"}
    )
    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )
    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-large-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    method_type: str = field(
        default='prefix',
        metadata={
            "help": "The method used for training: prefix, prompt, finetune, prompt or prefix -infill"
        }
    )
    # prefix: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Will use P-tuning v2 during training"
    #     }
    # )
    # prompt: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Will use prompt tuning during training"
    #     }
    # )
    # finetune: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Will use fine tuning during training"
    #     }
    # )
    pre_seq_len: int = field(
        default=20,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    )

    freeze_non_prompt_layers: int = field(
        default=0,
        metadata={
            "help": "whether freeze_non_prompt_layers should be 1 (to freeze) or 0 (not freeze) the non-prompt layers (e.g., lm_head)"
        }
    )
    shift_mask_pos_by_p_len: int = field(
        default=1,
        metadata={
            "help": "whether shift_mask_pos_by_p_len should be 1 (to shift) or 0 (not shift) for prompt-infill only"
        }
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    # training_type: str = field(
    #     default='public',
    #     # default='private',
    #     metadata={
    #         "help": "The type of training."
    #     }
    # )
    # dp_max_grad_norm: float = field(
    #     default=0.1,
    # )
    # dp_delta: float = field(
    #     default=1e-5,
    # )
    # dp_epsilon: float = field(
    #     default=7.5,
    # )


@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `version_2_with_negative=True`."
        },
    )


@dataclass
class PrivacyArgumentsCustom(dp_transformers.PrivacyArguments):
    per_sample_max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=1.0, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=1e-5, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=None, metadata={
        "help": "Target delta, defaults to 1/N"
    })
    disable_dp: bool = field(default=False, metadata={
        "help": "Disable DP training."
    })
    accounting_mode: str = field(
        default="rdp", metadata={"help": "One of (`rdp`, `glw`, `all`)."}
    )
    clipping_mode: str = field(
        default="default"
    )
    non_private: str = field(
        default="False", metadata={"help": "Train non-privately if True."}
    )

    use_discrete: int = field(
        default=1, metadata={"help": "use discrete gaussian noise or not "}
    )

    use_our_approximated_discrete_gaussian: int = field(
        default=0, metadata={"help": "use our discrete gaussian or not "}
    )

    discretization_granularity: float = field(
        default=0.0001, metadata={"help": "gamma in discretization"}
    )

    mod_num_bits: int = field(
        default=16, metadata={"help": "num bits used in mod "}
    )

    k_stddevs: int = field(
        default=4, metadata={"help": "num std devs"}
    )

    stochastic_rounding: int = field(
        default=1, metadata={"help": "whether to use stochastic rounding"}
    )
    beta: float = field(
        default=np.exp(-0.5), metadata={"help": "beta for conditional rounding. Set beta = 0 to disable the conditioning"}
    )

    num_colluding_clients: int = field(
        default=0, metadata={"help": ">1 iff when distributed DP"}
    )

    distributed_dp: int = field(
        default=0, metadata={"help": "whether to use distributed DP"}
    )


    def __post_init__(self):
        true_tags = ('y', 'yes', 't', 'true')
        self.non_private = self.non_private.lower() in true_tags  # noqa


@dataclass
class AuxiliaryArguments:
    eval_spectrum: str = field(default="no")
    max_spectrum_batches: int = field(default=100)
    max_lanczos_iter: int = field(default=100)

    store_grads: str = field(default="no")
    orthogonal_projection_path: Optional[str] = field(default=None)
    orthogonal_projection_rank: int = field(default=100)

    def __post_init__(self):
        true_tags = ('y', 'yes', 't', 'true')
        self.eval_spectrum = self.eval_spectrum.lower() in true_tags  # noqa
        self.store_grads = self.store_grads.lower() in true_tags  # noqa


def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArgumentsCustom,
                               QuestionAnwseringArguments, PrivacyArgumentsCustom, AuxiliaryArguments, CollaborativeLearningArguments))

    args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args = args
    data_args.infill = "infill" in model_args.method_type
    client_training_args = [copy.deepcopy(training_args) for _ in range(collaboration_args.num_clients)]
    client_privacy_args = [copy.deepcopy(privacy_args) for _ in range(collaboration_args.num_clients)]
    collaboration_args.client_data_split_ratio = [eval(i) for i in collaboration_args.client_data_split_ratio.split(",")]
    args = model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args
    return args
