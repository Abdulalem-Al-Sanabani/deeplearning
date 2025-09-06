import torch

from exercise_utils.core import BaseTrainer, make_summary_writer
from models.nmt import Seq2Seq
from exercise_utils.nlp.nmt.utils import (
    create_nmt_dataloaders, compute_bleu, TranslationExamplesMixin
)


class NMTTrainer(BaseTrainer, TranslationExamplesMixin):
    def __init__(self, model, config):
        self.src_tokenizer = config.src_tokenizer
        self.tgt_tokenizer = config.tgt_tokenizer
        train_loader, val_loader = create_nmt_dataloaders(config)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.max_steps
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
            scheduler=scheduler,
            config=config,
        )
        TranslationExamplesMixin.is_usable(self)

    def training_step(self, batch):
        loss = self.model.compute_loss(*batch)
        return {"loss": loss}

    def validation_step(self, batch):
        loss = self.model.compute_loss(*batch)
        return {"loss": loss}

    def run_experiment(self):
        for step, _, _ in self.fit():
            if step % self.config.eval_every_n_steps == 0:
                self.model.eval()
                # Compute BLEU score
                bleu_score = compute_bleu(
                    self.model,
                    self.val_loader,
                    self.model.src_tokenizer,
                    self.model.tgt_tokenizer,
                )
                self.logger.add_scalar(
                    f"{self.config.exp_name}/val_bleu", bleu_score, global_step=step
                )

                # add some fixed and random translation examples
                self.log_translation_examples(step)
