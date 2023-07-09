from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union
from my_enums import ContextType, Steps
import utils

if TYPE_CHECKING:
    from configs.trainer_config import TrainerConfig

    # from configs.inference_config import InferenceConfig
    from configs.dm_config import DataModuleConfig
    from base_datamodule import StepData


class DataModuleConfig:
    def __init__(
        self,
        num_workers=8,
        batch_size=32,
        eval_batch_size=32,
        test_batch_size=32,
        data_split_percent: list[float] = None,
        project_root: str = None,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        data_prep_out_root: str = "processed_data/simple_tod",
        max_token_len: int = 1024,
        test_prompt_max_len: int = 800,
        num_dialogs: list[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = False,
        overwrite: list[bool] = None,
        num_turns: int = 26,
        train_domain_settings: Union[list[str], str] = None,
        dev_domain_settings: Union[list[str], str] = None,
        test_domain_settings: Union[list[str], str] = None,
        train_domain_percentage: float = 1.0,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
        data_prep_multi_process: bool = True,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        train_step_data: "StepData" = None,
    ):
        self.num_workers = num_workers
        self.preprocessing_model_name = preprocessing_model_name
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / data_prep_out_root
        self.raw_data_root = raw_data_root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.max_token_len = max_token_len
        self.test_prompt_max_len = test_prompt_max_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name
        self.datasets: any = {}
        self.tokenizer = tokenizer or utils.get_tokenizer()
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.train_domain_percentage = train_domain_percentage
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.should_add_dsts = should_add_dsts

        self.data_prep_multi_process = data_prep_multi_process
        self.train_domain_settings = train_domain_settings
        self.dev_domain_settings = dev_domain_settings
        self.test_domain_settings = test_domain_settings
        # these two variables are added so that we can have typing in DataPrepConfig.from_dm_config method
        self.step_name = None
        self.domain_setting = None
        self.test_num_turns_groups = test_num_turns_groups
        self.train_step_data = train_step_data

    @classmethod
    def from_trainer_config(
        self,
        trainer_config: "TrainerConfig",
    ) -> "DataModuleConfig":
        return self(
            num_workers=trainer_config.num_workers,
            project_root=trainer_config.project_root,
            raw_data_root=trainer_config.raw_data_root,
            data_prep_out_root=trainer_config.data_prep_out_root,
            max_token_len=trainer_config.max_token_len,
            test_prompt_max_len=trainer_config.test_prompt_max_len,
            num_dialogs=trainer_config.num_dialogs,
            delexicalize=trainer_config.delexicalize,
            overwrite=trainer_config.overwrite,
            num_turns=trainer_config.num_turns,
            should_add_schema=trainer_config.should_add_schema,
            train_domain_settings=trainer_config.train_domain_settings,
            dev_domain_settings=trainer_config.dev_domain_settings,
            test_domain_settings=trainer_config.test_domain_settings,
            train_domain_percentage=trainer_config.train_domain_percentage,
            tokenizer=trainer_config.tokenizer,
            batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            data_split_percent=trainer_config.data_split_percent,
            should_add_user_actions=trainer_config.should_add_user_actions,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
            context_type=trainer_config.context_type,
            should_add_service_results=trainer_config.should_add_service_results,
            data_prep_multi_process=trainer_config.data_prep_multi_process,
            test_num_turns_groups=trainer_config.test_num_turns_groups,
        )
