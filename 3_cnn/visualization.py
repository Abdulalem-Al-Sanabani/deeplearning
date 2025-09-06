import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
from metrics import calculate_miou


def get_label_dict():
    return {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
        255: None,
    }


def create_custom_colormap(config):
    num_classes = config.num_classes  # Typically 21 (0 to 20)

    # Create a color array with num_classes + 1 colors (additional for void)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_classes + 1))

    # Assign a specific color
    colors[0] = [0, 0, 0.3, 1]  # Dark blue for background
    colors[-1] = [0.5, 0.5, 0.5, 1]  # Gray for void label

    # Create colormap
    cmap = mcolors.ListedColormap(colors)

    # Create normalizer
    bounds = np.arange(num_classes + 2) - 0.5  # +2 to include void
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def sample_image_mask_prediction(model, loader, config):
    model.eval()
    images, masks, predictions = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            if len(images) >= config.visualization_samples:
                break
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)

            images.extend(inputs.cpu())
            masks.extend(targets.cpu())
            predictions.extend(pred.cpu())

    return (
        images[: config.visualization_samples],
        masks[: config.visualization_samples],
        predictions[: config.visualization_samples],
    )


def unnormalize_image(img, mean, std):
    assert img.shape[0] == 3, "Image should be in CHW format"
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    return img


def image_torch2np(image, config):
    """Convert training image from torch tensor to plottable numpy array"""
    assert image.dim() == 3, "Make sure the input image is 3D tensor"
    image = unnormalize_image(image, config.preprocess_mean, config.preprocess_std)
    image_np = image.detach().permute(1, 2, 0).cpu().numpy()
    image_np = (255 * image_np).astype(np.uint8)
    return image_np


def mask_torch2np(mask):
    """Convert mask from torch tensor to numpy array"""
    assert mask.dim() == 2, "Make sure the input mask is 2D tensor"
    return mask.detach().cpu().numpy().astype(np.int64)


def find_medoid(X, num_samples=200):
    # X.shape = (N, 2)
    N = X.shape[0]

    # Sample a subset of points if N > num_samples
    if N > num_samples:
        sample_indices = np.random.choice(N, num_samples, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    diff = X_sample[:, np.newaxis, :] - X[np.newaxis, :, :]  # (sample_size, N, 2)
    dist = np.sqrt(np.sum(diff**2, axis=-1))  # (sample_size, N)

    total_dist = np.sum(dist, axis=0)  # (N,)

    medoid_index = np.argmin(total_dist)
    medoid = X[medoid_index]  # (2,)
    return medoid


def determine_label_positions(
    pred, eps=10, min_samples=50, num_samples=200, include_noise=True
):
    """
    Determine label locations for each class in a semantic segmentation prediction
    Based on density-based clustering and medoids

    Args:
        pred: np.array of semantic class prediction (img_size, img_size)
        eps: DBSCAN hyperparameter
        min_samples: DBSCAN hyperparameter
        num_samples: number of samples to use for finding medoids for each cluster in each class
        include_noise:
            If False, exclude noisy points from the results
            If True, every unique class present in the prediction will be given at least one label position. This is useful for debugging, e.g., mIoU calculation.

    Returns:
        label_pos: list[tuple[str, tuple[int, int]]]
            Label names and their determined (y, x) positions.
            e.g., [('chair': (201, 24)), ('diningtable', (185, 56)), ...]
    """
    label_dict = get_label_dict()
    unique_classes = sorted(np.unique(pred))

    assert label_dict[0] == "background"
    unique_classes = [idx for idx in unique_classes if idx != 0 and idx != 255]

    label_pos = []

    # for each class in a mask prediction
    for cls_index in unique_classes:
        # step 1: get pixel xy values of the class
        y, x = np.where(pred == cls_index)
        pixel_xy = np.column_stack([x, y])

        # step 2: density-based clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pixel_xy)
        cluster_labels = clustering.labels_
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label != -1:  # Exclude noise points (labeled as -1)
                cluster_points = pixel_xy[cluster_labels == label]

                # step 3: find medoids for clusters
                medoid = find_medoid(cluster_points, num_samples)

                # append to returns
                cls_name = label_dict[cls_index]
                label_pos.append((cls_name, (medoid[0], medoid[1])))

        # sample one point noisy class to assign a label if include_noise is True
        if len(unique_labels) == 1 and unique_labels[0] == -1 and include_noise:
            cluster_points = pixel_xy[cluster_labels == -1]
            sampled_point = cluster_points[np.random.choice(cluster_points.shape[0])]
            cls_name = label_dict[cls_index]
            label_pos.append((cls_name, (sampled_point[0], sampled_point[1])))

    return label_pos


def add_labels_to_image(ax, mask):
    label_pos = determine_label_positions(mask)
    for cls_name, xy in label_pos:
        x, y = xy
        ax.text(
            x,
            y,
            cls_name,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.7, pad=1),
        )


def mask2rgb(mask, config):
    color_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)

    # Assign different colors to different classes
    cmap, _ = create_custom_colormap(config)
    colors = {i: 255 * color[:3] for i, color in enumerate(cmap.colors)}

    # Replace void label with the last color
    colors[config.void_label] = colors[config.num_classes]
    colors.pop(config.num_classes)

    for class_id, color in colors.items():
        color_mask[mask == class_id] = color.reshape(1, 1, 3)

    return color_mask


def composite_mask_and_image(
    image: np.ndarray, mask: np.ndarray, config, draw_background=True
):
    """Alpha compositing mask on top of the image

    Args:
        image: np.ndarray of shape (H, W, 3)
        mask: np.ndarray of shape (H, W)

    Returns:
        blended: np.ndarray of shape (H, W, 3)
    """
    alpha = config.alpha

    # Convert mask to RGB
    color_mask = mask2rgb(mask, config)

    if draw_background:
        blended = alpha * color_mask + (1 - alpha) * image
    else:
        blended = np.where(
            np.expand_dims(mask == 0, axis=-1),  # mask == 0 is background
            image,
            (alpha * color_mask + (1 - alpha) * image),
        )

    blended = blended.astype(np.uint8)
    return blended


def visualize_segmentation(images, masks, predictions, config):
    fig, axes = plt.subplots(
        3, config.visualization_samples, figsize=(3 * config.visualization_samples, 9)
    )

    for i in range(config.visualization_samples):
        image_np = image_torch2np(images[i], config)
        mask_np = mask_torch2np(masks[i].squeeze())
        pred_np = mask_torch2np(predictions[i].squeeze())

        # Original image
        axes[0, i].imshow(image_np)
        axes[0, i].set_title("Original Image")
        axes[0, i].axis("off")

        # Blended ground truth
        blended_gt = composite_mask_and_image(image_np, mask_np, config)
        axes[1, i].imshow(blended_gt)
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")
        add_labels_to_image(axes[1, i], mask_np)

        # Blended prediction
        blended_pred = composite_mask_and_image(image_np, pred_np, config)
        axes[2, i].imshow(blended_pred)
        axes[2, i].axis("off")
        add_labels_to_image(axes[2, i], pred_np)
        # also add mIoU to the title
        miou = calculate_miou(
            torch.from_numpy(pred_np),
            torch.from_numpy(mask_np),
            config.num_classes,
            config.void_label,
        )
        axes[2, i].set_title(f"Prediction | mIoU: {miou:.3f}")

    plt.tight_layout()
    return fig
