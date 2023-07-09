from pathlib import Path
from omegaconf import DictConfig
import hydra
import torch
import gc
from transformers import (
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from datamodules.base_datamodule import BaseDataModule
from configs.dm_config import DataModuleConfig

from configs.trainer_config import TrainerConfig

from inference import Inference
from datamodules.tod_datamodules import TodDataModule
import os
import warnings
import my_enums
import utils
from zs_tod.src.configs.inference_config import InferenceConfig

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"] = "INFO"
import wandb
from my_enums import Steps, TrainingStage


class ZsToDTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config

    def print_cuda_info(self, step=""):
        if step:
            print(f"Step: {step}")
        print(torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_cached() / 1024**2)

    def run(self):
        self.print_cuda_info("init")
        current_dir = Path(os.getcwd())
        print(str(current_dir))

        self.cfg.datamodule = self._get_dm()
        if self.cfg.train_model_path:
            pretrained_model_path = str(
                self.cfg.project_root / self.cfg.train_model_path
            )
        else:
            pretrained_model_path = self.pretrain_model(self.cfg.datamodule)
        self.print_cuda_info("after pretrain")
        gc.collect()

        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before training")
        if self.cfg.two_step_training:
            out_dir = self.train_model(pretrained_model_path, self.cfg.datamodule)
            full_out_dir = str(current_dir / out_dir)
        else:
            full_out_dir = str(current_dir / pretrained_model_path)
        self.print_cuda_info("after train")
        print("Training done")
        print("-" * 80)
        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before testing")
        if self.cfg.should_test:
            inf = Inference(
                InferenceConfig.from_trainer_config(self.cfg, full_out_dir),
            )
            inf.test()
        print(full_out_dir)

    def _get_dm(self) -> BaseDataModule:
        steps = Steps.list() if self.cfg.should_test else Steps.list()[:-1]
        return TodDataModule(DataModuleConfig.from_trainer_config(self.cfg), steps)

    def _get_trainer(
        self,
        model_train: AutoModel,
        dm: TodDataModule,
        training_args: TrainingArguments,
        training_stage: TrainingStage = TrainingStage.TRAIN,
    ) -> Trainer:
        collator = (
            dm.training_collator
            if training_stage == TrainingStage.TRAIN
            else dm.pretraining_collator
        )
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.datasets[my_enums.Steps.TRAIN.value],
            eval_dataset=dm.datasets[my_enums.Steps.DEV.value],
            data_collator=collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                ),
            ],
        )
        return trainer

    def _get_training_args(
        self, step_name: str, epochs: int, train_batch_size: int
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.cfg.out_dir / step_name),
            num_train_epochs=epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=5,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.cfg.eval_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            report_to="wandb",
            fp16=self.cfg.fp16,
            dataloader_drop_last=True,
            run_name=step_name,
            learning_rate=5e-4,
        )

    def get_model_instance(self, path: str = None) -> AutoModel:
        model_class = utils.get_model_class(self.cfg.model_name)
        model = model_class.from_pretrained(path or self.cfg.model_name)
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        return model

    def pretrain_model(self, dm: TodDataModule) -> str:
        if self.cfg.pretrain_model_path:
            path = self.cfg.project_root / self.cfg.pretrain_model_path
            if path.exists():
                return str(path)
        training_args = self._get_training_args(
            "pretrain", self.cfg.pretrain_epochs, self.cfg.pretrain_batch_size
        )
        model = self.get_model_instance()

        pre_trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.PRETRAIN
        )
        model.config.use_cache = False
        model.train()
        pre_trainer.train()
        pre_trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        torch.cuda.empty_cache()
        return training_args.output_dir

    def train_model(self, path, dm) -> str:
        model = self.get_model_instance(path)
        training_args = self._get_training_args(
            "train", self.cfg.train_epochs, self.cfg.train_batch_size
        )
        trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.TRAIN
        )
        model.train()
        trainer.train()
        trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        out_dir = os.getcwd()
        print("training output_dir: ", out_dir)
        return training_args.output_dir


@hydra.main(config_path="../configs/trainer/", config_name="zs_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    trainer_cfg = TrainerConfig(**cfg)
    # utils.init_wandb(trainer_cfg, cfg, "training")
    stt = ZsToDTrainer(trainer_cfg)
    stt.run()


if __name__ == "__main__":
    hydra_start()
