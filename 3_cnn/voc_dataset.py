import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.datasets import VOCSegmentation


class ToTensor(nn.Module):
    """
    Convert image and mask from PIL.Image to torch.Tensor (float32 and int64 respectively)
    """

    def __init__(self):
        super().__init__()

    def forward(self, img, mask):
        assert isinstance(
            img, Image.Image
        ), f"Image must be a PIL image, get {type(img)}"
        assert isinstance(
            mask, Image.Image
        ), f"Mask must be a PIL image, get {type(mask)}"
        assert (
            img.size == mask.size
        ), f"Image and mask must have the same shape, get {img.size} and {mask.size}"

        img_tensor = TF.to_tensor(img).to(torch.float32)
        mask_tensor = (255 * TF.to_tensor(mask)).to(torch.int64)

        assert (
            img_tensor.dtype == torch.float32
        ), f"Image must be float32, get {img_tensor.dtype}"
        assert (
            mask_tensor.dtype == torch.int64
        ), f"Mask must be int64, get {mask_tensor.dtype}"

        return img_tensor, mask_tensor


class PadTo(nn.Module):
    """
    If the image is smaller than the shape, pad it to the shape
    """

    def __init__(self, shape, fill=0):
        super().__init__()
        self.shape = shape  # (h, w). pad to this shape
        self.fill = fill

    def forward(self, img, mask):
        assert isinstance(img, Image.Image), "Image must be a PIL image"
        assert img.size == mask.size, "Image and mask must have the same shape"

        w, h = img.size
        if h >= self.shape[0] and w >= self.shape[1]:
            return img, mask

        img_tensor, mask_tensor = self._Image2Tensor(img, mask)

        pad_h = max(self.shape[0] - h, 0)
        pad_w = max(self.shape[1] - w, 0)
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]

        img_padded = TF.pad(img_tensor, padding, fill=self.fill)
        mask_padded = TF.pad(mask_tensor, padding, fill=self.fill)

        return self._Tensor2Image(img_padded, mask_padded)

    def _Image2Tensor(self, img, mask):
        return TF.to_tensor(img), TF.to_tensor(mask)

    def _Tensor2Image(self, img, mask):
        return TF.to_pil_image(img), TF.to_pil_image(mask)


class VOCDataset(Dataset):
    """
    A custom VOC segmentation dataset with (similar) transformations/augmentations used by DeepLabV3.

    train_transform:
        RandomScale (0.5~2) + RandomCrop + RandomHorizontalFlip + Normalize

    val_transform:
        CenterCrop + Normalize
    """

    def __init__(self, image_set, config, root="../datasets"):
        assert image_set in [
            "train",
            "val",
        ], "Image set must be either 'train' or 'val'"
        self.root = root
        self.image_set = image_set
        self.config = config
        image_shape = (config.img_size, config.img_size)

        # raw data
        self.dataset = self._load_data()

        self.train_transform = v2.Compose(
            [
                v2.RandomResize(
                    int(self.config.img_size * 0.5), int(self.config.img_size * 2)
                ),
                PadTo(image_shape),
                v2.RandomCrop(image_shape),
                v2.RandomHorizontalFlip(),
                ToTensor(),
                v2.Normalize(
                    mean=self.config.preprocess_mean, std=self.config.preprocess_std
                ),
            ]
        )

        self.val_transform = v2.Compose(
            [
                v2.Resize(
                    self.config.img_size
                ),  # Scale min(H, W) to img_size while keeping aspect ratio
                v2.CenterCrop(image_shape),
                ToTensor(),
                v2.Normalize(
                    mean=self.config.preprocess_mean, std=self.config.preprocess_std
                ),
            ]
        )

    def _load_data(self):
        voc_root = os.path.join(self.root, "VOCdevkit", "VOC2012")
        if os.path.exists(voc_root):
            download = False
        else:
            print("VOC dataset not found. Downloading and extracting...")
            download = True

        dataset = VOCSegmentation(
            root=self.root,
            year="2012",
            image_set=self.image_set,
            download=download,
        )

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        if self.image_set == "train":
            return self.train_transform(img, mask)
        else:
            return self.val_transform(img, mask)
