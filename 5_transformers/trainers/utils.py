import torch.optim as optim


def create_optimizer(model, config):
    return optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )


def create_scheduler(optimizer, config):
    if config.scheduler == "cosine":
        return optim.lr_scheduler.ChainedScheduler(
            [
                # Linear warmup
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # start from 1% of base lr
                    end_factor=1.0,
                    total_iters=config.warmup_steps,
                ),
                # Cosine decay
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps),
            ]
        )
    else:
        return None
