import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.mini_lm import MiniLM
from exercise_utils.core import BaseTrainer, make_summary_writer
from exercise_utils.nlp.lm.utils import (
    create_lm_dataloaders,
    generate_text,
    format_generation_logging,
)


class LMTrainer(BaseTrainer):
    def __init__(self, model, config):
        self.tokenizer = config.tokenizer
        model = MiniLM(model, config)
        train_loader, val_loader = create_lm_dataloaders(config, config.seq_len)
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
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
            config=config,
        )

    def training_step(self, batch):
        """
        Perform a single training step on a batch of data

        Args:
            batch: Tuple containing input and target tensors
                inputs: Input tensor of shape (batch_size, seq_len) containing token indices
                targets: Target tensor of shape (batch_size, seq_len) containing token indices

        Returns:
            Dict containing the loss and perplexity for the batch
        """
        loss = self.model(*batch)
        perplexity = torch.exp(loss)
        return {"loss": loss, "perplexity": perplexity}

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for step, train_metrics, val_metrics in self.fit():
            # Generate some text with and without a prompt
            prompts = ["Now is the winter of our discontent", None]
            text = ""
            for prompt in prompts:
                text_gen = generate_text(
                    self.model, self.tokenizer, self.config.generate_text_length, prompt
                )
                text_gen = format_generation_logging(text_gen, prompt)
                text += text_gen
            self.logger.add_text("Generated Text", text, step)
