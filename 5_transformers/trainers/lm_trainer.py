import torch
from models import GPT

from .utils import create_optimizer, create_scheduler
from exercise_utils.core import BaseTrainer, make_summary_writer
from exercise_utils.nlp.lm.utils import (
    create_lm_dataloaders,
    generate_text,
    format_generation_logging,
)


def generate(model: GPT, tokenizer, prompt: str, max_new_tokens: int = 500):
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    token_ids = model.generate(token_ids, max_new_tokens)
    return tokenizer.decode(token_ids.squeeze().tolist())


class LMTrainer(BaseTrainer):
    def __init__(self, model, config):
        self.tokenizer = config.tokenizer
        train_loader, val_loader = create_lm_dataloaders(
            config, config.transformer_config.context_length
        )
        model = model.to(config.device)
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)

        logger = make_summary_writer(config.exp_name, config.config_name)

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            max_steps=config.max_steps,
            eval_every_n_steps=config.eval_every_n_steps,
            logger=logger,
            scheduler=scheduler,
            config=config,
        )

    def training_step(self, batch):
        loss = self.model.compute_loss(*batch)
        perplexity = torch.exp(loss)
        return {"loss": loss, "perplexity": perplexity}

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for step, _, _ in self.fit():
            if step % self.config.eval_every_n_steps == 0:
                # Generate some text with and without a prompt
                prompts = ["Now is the winter of our discontent", None]
                text = ""
                for prompt in prompts:
                    text_gen = generate_text(
                        self.model,
                        self.tokenizer,
                        self.config.generate_text_length,
                        prompt,
                    )
                    text_gen = format_generation_logging(text_gen, prompt)
                    text += text_gen
                self.logger.add_text("Generated Text", text, step)
