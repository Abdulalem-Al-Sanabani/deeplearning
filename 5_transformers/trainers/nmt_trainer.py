from models.transformer import Transformer
from .utils import create_optimizer, create_scheduler
from exercise_utils.core import make_summary_writer, BaseTrainer
from exercise_utils.nlp.nmt.utils import (
    TranslationExamplesMixin,
    compute_bleu,
    create_nmt_dataloaders,
)


class NMTTrainer(BaseTrainer, TranslationExamplesMixin):
    def __init__(self, model, config):
        self.src_tokenizer = config.src_tokenizer
        self.tgt_tokenizer = config.tgt_tokenizer
        model = model.to(config.device)
        optimizer = create_optimizer(model, config)
        train_loader, val_loader = create_nmt_dataloaders(config)
        scheduler = create_scheduler(optimizer, config)

        if config.log_to_tensorboard:
            logger = make_summary_writer(config.exp_name, config.config_name)
        else:
            logger = None

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
        loss = self.model(*batch)
        return {"loss": loss}

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for step, _, _ in self.fit():
            if self.logger and step % self.config.eval_every_n_steps == 0:
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
