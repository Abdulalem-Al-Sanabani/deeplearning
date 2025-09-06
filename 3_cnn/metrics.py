import torch
import torch.nn.functional as F


def calculate_miou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, void_label: int
) -> torch.Tensor:
    """
    Calculate the mean Intersection over Union (mIoU) for semantic segmentation.
    Reference: https://medium.com/@cyborg.team.nitr/miou-calculation-4875f918f4cb

    Args:
        pred: int tensor (of arbitrary shape) containing the predicted labels (from 0 to num_classes - 1)
        target: int tensor containing the ground truth labels (from 0 to num_classes - 1)
        num_classes: number of classes
        void_label: label to ignore in loss calculation and mIoU

    Returns:
        mIoU: scalar tensor of the mean Intersection over Union
    """
    assert pred.size() == target.size(), "pred and target must have the same shape"
    pred = pred.flatten().cpu()
    target = target.flatten().cpu()

    # mask out void label
    void_mask = target == void_label
    pred = pred[~void_mask]
    target = target[~void_mask]

    assert torch.isin(
        pred, torch.arange(num_classes)
    ).all(), (
        "Make sure prediction values are natural number in {0, ..., num_classes - 1}"
    )
    assert torch.isin(
        target, torch.arange(num_classes)
    ).all(), "Make sure label values are natural number in {0, ..., num_classes - 1}"

    # construct confusion matrix
    category = target * num_classes + pred
    confusion_matrix = torch.bincount(category, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )

    # compute TP, FP, FN per class
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp

    # compute IoU per class
    iou_per_class = tp / (tp + fp + fn)

    # compute mIoU
    miou = torch.nanmean(iou_per_class)
    return miou
