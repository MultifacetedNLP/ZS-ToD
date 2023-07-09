import utils
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from base_datamodule import BaseDataModule
from pathlib import Path
from my_enums import ContextType


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d1/adibm-data/projects/ZSToD",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        model_name: str = "gpt2",
        tokenizer_name: str = None,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        early_stopping_patience: int = 3,
        eval_steps: int = 500,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        train_batch_size: int = 8,
        pretrain_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_percentage: float = 1.0,
        train_domain_settings: list[str] = None,
        dev_domain_settings: list[str] = None,
        test_domain_settings: list[list[str]] = None,
        out_dir: str = "results",
        pretrain_epochs: int = 1,
        pretrain_model_path: str = None,
        train_model_path: str = None,
        train_epochs: int = 1,
        logging_dir: str = "logs",
        generate_max_len: int = 1024,
        should_test: bool = False,
        logging_steps: int = 50,
        test_prompt_max_len: int = 799,
        max_token_len: int = 1024,
        eval_accumulation_steps: int = 16,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
        fp16: int = False,
        postprocess_generation: bool = False,
        wandb: any = None,
        data_prep_multi_process: bool = True,
        datamodule: "BaseDataModule" = None,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        two_step_training: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.early_stopping_patience = early_stopping_patience
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs or [20, 10, 17]
        self.delexicalize = delexicalize
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = Path(out_dir)
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.dev_domain_settings = dev_domain_settings or ["seen"]
        self.train_domain_settings = train_domain_settings or ["seen"]
        self.test_domain_settings = test_domain_settings or [
            ["all"],
            ["seen"],
            ["unseen"],
        ]
        self.train_domain_percentage = train_domain_percentage
        self.pretrain_model_path = pretrain_model_path
        self.train_model_path = train_model_path
        self.logging_dir = Path(logging_dir)
        self.generate_max_len = generate_max_len
        self.should_test = should_test
        self.delexicalize = delexicalize
        self.logging_steps = logging_steps
        self.train_batch_size = train_batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.raw_data_root = self.project_root / raw_data_root
        self.test_prompt_max_len = test_prompt_max_len
        self.eval_accumulation_steps = eval_accumulation_steps
        self.fp16 = fp16

        self.tokenizer_name = tokenizer_name or model_name
        self.tokenizer = utils.get_tokenizer(self.tokenizer_name)
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        if test_prompt_max_len > max_token_len:
            raise ValueError("context_max_len must be less than max_token_len")
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results

        self.should_add_dsts = should_add_dsts
        self.postprocess_generation = postprocess_generation
        self.data_prep_multi_process = data_prep_multi_process
        self.wandb = wandb
        self.test_num_turns_groups = test_num_turns_groups
        self.datamodule = datamodule
        self.two_step_training = two_step_training
