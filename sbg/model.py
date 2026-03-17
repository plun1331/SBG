"""
ML model for Super Battle Golf shot prediction.

Defines a convolutional neural network (CNN) that accepts either a raw game
screen image or a pre-computed feature vector and outputs the recommended
shot parameters (direction, power, loft).

Architecture
------------
* **Image branch** – A lightweight CNN (3 convolutional blocks with BatchNorm
  and max-pooling) that encodes the 3-channel input image into a 256-dim
  embedding.
* **Feature branch** – A two-layer MLP that encodes the 11-dim game-state
  feature vector into a 64-dim embedding.
* **Head** – Concatenation of both embeddings fed through a fully-connected
  head with dropout regularisation, producing three scalar outputs:

    * ``direction_deg`` – aimed direction offset in degrees (tanh × 45°).
    * ``power``         – shot power in [0, 1] (sigmoid).
    * ``loft_deg``      – loft angle in degrees (sigmoid × 90°).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimension of the feature vector produced by GameState.to_feature_vector()
# Breakdown: 11 hit-bar values
FEATURE_DIM = 11

# Default image input size fed to the CNN branch
DEFAULT_IMG_H = 120
DEFAULT_IMG_W = 160


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShotPredictorModel(nn.Module):
    """
    Dual-branch neural network for predicting golf-shot parameters.

    Parameters:
        img_h: Expected input image height (pixels).
        img_w: Expected input image width (pixels).
        feature_dim: Dimensionality of the game-state feature vector.
        dropout: Dropout probability applied in the prediction head.
    """

    def __init__(
        self,
        img_h: int = DEFAULT_IMG_H,
        img_w: int = DEFAULT_IMG_W,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ---- Image branch (CNN) ----------------------------------------
        self.cnn = nn.Sequential(
            _ConvBlock(3, 32),    # → H/2,  W/2,  32
            _ConvBlock(32, 64),   # → H/4,  W/4,  64
            _ConvBlock(64, 128),  # → H/8,  W/8,  128
            _ConvBlock(128, 256), # → H/16, W/16, 256
        )
        cnn_out_h = img_h // 16
        cnn_out_w = img_w // 16
        cnn_flat_dim = 256 * cnn_out_h * cnn_out_w

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- Feature branch (MLP) -------------------------------------
        self.feat_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ---- Prediction head ------------------------------------------
        combined_dim = 256 + 64
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # direction, power, loft
        )

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters:
            image: Float tensor of shape ``(B, 3, H, W)`` with values in
                ``[0, 1]``.
            features: Float tensor of shape ``(B, feature_dim)``.

        Returns:
            Tuple ``(direction_deg, power, loft_deg)`` each of shape ``(B,)``.

            * ``direction_deg``: degrees in ``[-45, 45]``  (tanh activation).
            * ``power``        : value in ``[0, 1]``       (sigmoid activation).
            * ``loft_deg``     : degrees in ``[0, 90]``    (sigmoid × 90).
        """
        img_emb = self.cnn_fc(self.cnn(image))
        feat_emb = self.feat_fc(features)
        combined = torch.cat([img_emb, feat_emb], dim=1)
        out = self.head(combined)

        direction_deg = torch.tanh(out[:, 0]) * 45.0
        power = torch.sigmoid(out[:, 1])
        loft_deg = torch.sigmoid(out[:, 2]) * 90.0

        return direction_deg, power, loft_deg

    def predict(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Convenience wrapper for single-sample inference (no gradient).

        Parameters:
            image: Float tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.
            features: Float tensor of shape ``(1, F)`` or ``(F,)``.

        Returns:
            Tuple ``(direction_deg, power, loft_deg)`` as plain Python floats.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            direction, power, loft = self.forward(image, features)

        return float(direction[0]), float(power[0]), float(loft[0])

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "ShotPredictorModel":
        """
        Load a previously saved model from *path*.

        Parameters:
            path: Path to the saved state-dict file.
            device: Target device (defaults to CPU).
            **kwargs: Forwarded to the constructor.
        """
        model = cls(**kwargs)
        state = torch.load(path, map_location=device or torch.device("cpu"))
        model.load_state_dict(state)
        return model
