from abc import ABC, abstractmethod
from typing import Union

from dotmap import DotMap
from my_enums import SpecialTokens
from transformers import AutoModel, AutoTokenizer
from torch import Tensor


class GenerationBase(ABC):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def _get_generation(self, batch, max_len: int) -> Union[Tensor, list[Tensor]]:
        pass

    def move_to_gpu(self, batch):
        batch_gpu = DotMap()
        batch_gpu.input_ids = batch.input_ids.cuda()
        batch_gpu.attention_masks = batch.attention_masks.cuda()
        return batch_gpu

    @abstractmethod
    def remove_context(self, gen: Union[Tensor, list[Tensor]], context_len: int):
        pass

    def get_generation(
        self,
        batch,
        max_len: int,
        context_len: int,
        should_post_process: bool,
    ) -> list[str]:
        batch_gpu = self.move_to_gpu(batch)
        gen = self._get_generation(batch_gpu, max_len)
        gen_without_context = self.remove_context(gen, context_len, max_len)
        gen_after_hook = self.hook_before_remove_pad(gen_without_context)
        gen_no_pad = self.remove_padding(gen_after_hook)
        gen_txt = self.tokenizer.batch_decode(gen_no_pad, skip_special_tokens=False)
        if should_post_process:
            return self.postprocess_generation(gen_txt)
        return gen_txt

    def hook_before_remove_pad(self, gen: Union[Tensor, list[Tensor]]):
        return gen

    def postprocess_generation(self, batch: list[str]) -> list[str]:
        out = []
        required_tokens = [
            SpecialTokens.begin_target,
            SpecialTokens.begin_dsts,
            SpecialTokens.begin_dst,
            SpecialTokens.begin_intent,
        ]
        for item in batch:
            text_to_add = [rt for rt in required_tokens if rt not in item]
            out_text = "".join(["".join(text_to_add), item])
            out.append(out_text)
        return out

    def remove_padding(self, gen):
        return [row[row != self.tokenizer.pad_token_id] for row in gen]
