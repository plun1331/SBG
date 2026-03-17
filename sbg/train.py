"""
Training utilities for the Super Battle Golf shot-prediction model.

Provides:
* ``ShotDataset``    – a PyTorch Dataset that loads (image, features, labels)
  tuples from a directory of .npz files.
* ``train_one_epoch`` – runs one epoch of supervised training.
* ``evaluate``        – computes mean loss on a validation split.
* ``train``           – full training loop with early stopping.

Data format
-----------
Each sample is stored in a NumPy ``.npz`` file with the following keys:

* ``image``     – uint8 array of shape ``(H, W, 3)`` (BGR, raw screen crop).
* ``features``  – float32 array of shape ``(11,)`` from
  ``GameState.to_feature_vector()``.
* ``direction`` – float scalar: ground-truth direction offset (degrees).
* ``power``     – float scalar: ground-truth power in [0, 1].
* ``loft``      – float scalar: ground-truth loft angle (degrees).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sbg.model import ShotPredictorModel, DEFAULT_IMG_H, DEFAULT_IMG_W


class ShotDataset(Dataset):
    """
    Dataset that loads shot samples from a directory of ``.npz`` files.

    Parameters:
        data_dir: Path to the directory containing ``.npz`` sample files.
        img_h: Target image height (pixels) – images are resized if needed.
        img_w: Target image width (pixels).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        img_h: int = DEFAULT_IMG_H,
        img_w: int = DEFAULT_IMG_W,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._img_h = img_h
        self._img_w = img_w
        self._files = sorted(Path(data_dir).glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No .npz files found in '{data_dir}'"
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self._files[idx])

        img = data["image"]  # H×W×3 uint8
        if img.shape[:2] != (self._img_h, self._img_w):
            img = self._cv2.resize(
                img, (self._img_w, self._img_h),
                interpolation=self._cv2.INTER_LINEAR
            )
        # BGR → RGB, normalise to [0, 1]
        img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        features = torch.from_numpy(data["features"].astype(np.float32))

        labels = torch.tensor(
            [
                float(data["direction"]),
                float(data["power"]),
                float(data["loft"]),
            ],
            dtype=torch.float32,
        )

        return {"image": image_tensor, "features": features, "labels": labels}


def _shot_loss(
    pred_direction: torch.Tensor,
    pred_power: torch.Tensor,
    pred_loft: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE loss combining direction, power, and loft targets."""
    target_dir = labels[:, 0]
    target_power = labels[:, 1]
    target_loft = labels[:, 2]

    loss_dir = nn.functional.mse_loss(pred_direction, target_dir)
    loss_power = nn.functional.mse_loss(pred_power, target_power)
    loss_loft = nn.functional.mse_loss(pred_loft / 90.0, target_loft / 90.0)

    # Weight direction most heavily as it has the greatest impact on accuracy
    return 2.0 * loss_dir + loss_power + loss_loft


def train_one_epoch(
    model: ShotPredictorModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        direction, power, loft = model(images, features)
        loss = _shot_loss(direction, power, loft, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: ShotPredictorModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean loss on a validation / test DataLoader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            direction, power, loft = model(images, features)
            loss = _shot_loss(direction, power, loft, labels)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train(
    data_dir: Union[str, Path],
    save_path: Union[str, Path],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    patience: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    img_h: int = DEFAULT_IMG_H,
    img_w: int = DEFAULT_IMG_W,
) -> ShotPredictorModel:
    """
    Full training loop for the shot-prediction model.

    Parameters:
        data_dir: Directory containing ``.npz`` training samples.
        save_path: Path where the best model weights will be saved.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for the Adam optimiser.
        val_split: Fraction of samples reserved for validation.
        patience: Early-stopping patience (epochs without improvement).
        device: Target compute device.
        img_h: Image height expected by the model.
        img_w: Image width expected by the model.

    Returns:
        The trained ShotPredictorModel (loaded with best weights).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    dataset = ShotDataset(data_dir, img_h=img_h, img_w=img_w)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = ShotPredictorModel(img_h=img_h, img_w=img_w).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 2, factor=0.5
    )

    best_val_loss = float("inf")
    no_improve = 0
    save_path = Path(save_path)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model.save(str(save_path))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # Reload best weights
    model = ShotPredictorModel.load(
        str(save_path), device=device, img_h=img_h, img_w=img_w
    )
    return model
