import torch
from torch.utils.data import DataLoader

from exercise_utils.core import make_summary_writer, BaseTrainer

from config import ExperimentConfig
from visualization import sample_image_mask_prediction, visualize_segmentation
from voc_dataset import VOCDataset
from semantic_segmentation import *


def create_dataloaders(config: ExperimentConfig):
    train_dataset = VOCDataset(root="../datasets", image_set="train", config=config)
    val_dataset = VOCDataset(root="../datasets", image_set="val", config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    return train_loader, val_loader


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=config.max_steps
        )
    elif config.lr_scheduler == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=config.max_steps, power=0.9
        )
    else:
        scheduler = None
    return scheduler


class CNNTrainer(BaseTrainer):
    def __init__(self, model, config: ExperimentConfig):
        self.model = model
        optimizer = self.model.configure_optimizers()
        train_loader, val_loader = create_dataloaders(config)
        logger = make_summary_writer(config.exp_name, config.config_name)
        scheduler = create_lr_scheduler(optimizer, config)

        super().__init__(
            model=self.model,
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
        inputs, targets = batch
        outputs = self.model(inputs)
        metrics = self.step_fn(outputs, targets, self.config.void_label)
        return metrics

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for step, _, _ in self.fit():
            if (
                step % self.config.eval_every_n_steps == 0
                or step == self.config.max_steps
            ):
                train_images, train_masks, train_preds = sample_image_mask_prediction(
                    self.model, self.train_loader, self.config
                )
                train_fig = visualize_segmentation(
                    train_images, train_masks, train_preds, self.config
                )
                self.logger.add_figure(f"Segmentation Result Training", train_fig, step)

                val_images, val_masks, val_preds = sample_image_mask_prediction(
                    self.model, self.val_loader, self.config
                )
                val_fig = visualize_segmentation(
                    val_images, val_masks, val_preds, self.config
                )
                self.logger.add_figure(f"Segmentation Result Validation", val_fig, step)

                self.logger.flush()
