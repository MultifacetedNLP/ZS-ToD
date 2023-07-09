from generation.simple_generation import SimpleGeneration


class GenerationHandlerFactory:
    @classmethod
    def get_handler(self, cfg):
        return SimpleGeneration(cfg.model, cfg.tokenizer)
