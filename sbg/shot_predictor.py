"""
High-level shot predictor for Super Battle Golf.

The ShotPredictor integrates ScreenAnalyzer and ShotPredictorModel into a
single, easy-to-use interface.  Given a raw game-screen frame it:

1. Runs the ScreenAnalyzer to produce a GameState.
2. Converts the GameState into a feature vector.
3. Pre-processes the frame into an image tensor.
4. Queries the ShotPredictorModel for shot parameters.
5. Returns a ShotParameters dataclass.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from sbg.game_state import GameState, ShotParameters
from sbg.model import ShotPredictorModel, DEFAULT_IMG_H, DEFAULT_IMG_W
from sbg.screen_analyzer import ScreenAnalyzer


class ShotPredictor:
    """
    End-to-end predictor: raw screen image → recommended shot parameters.

    Parameters:
        model_path: Optional path to a saved model weights file.  If not
            provided a randomly-initialised model is used (useful for testing
            and as a baseline before training).
        device: PyTorch device string or object (e.g. ``"cuda"`` or ``"cpu"``).
            Defaults to CUDA if available, otherwise CPU.
        img_h: Image height expected by the model (pixels).
        img_w: Image width expected by the model (pixels).
        screen_analyzer: ScreenAnalyzer instance to use.  A default instance
            is created if not provided.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        img_h: int = DEFAULT_IMG_H,
        img_w: int = DEFAULT_IMG_W,
        screen_analyzer: Optional[ScreenAnalyzer] = None,
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self._img_h = img_h
        self._img_w = img_w

        if model_path is not None:
            self._model = ShotPredictorModel.load(
                str(model_path), device=self.device,
                img_h=img_h, img_w=img_w,
            )
        else:
            self._model = ShotPredictorModel(img_h=img_h, img_w=img_w)
        self._model.to(self.device)
        self._model.eval()

        self._analyzer = screen_analyzer or ScreenAnalyzer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_from_frame(self, frame: np.ndarray) -> ShotParameters:
        """
        Predict the best shot given a raw game-screen frame.

        Parameters:
            frame: An H×W×3 BGR image (as returned by OpenCV, for example).

        Returns:
            ShotParameters with direction, power, and loft recommendations.
        """
        state = self._analyzer.analyze(frame)
        return self.predict_from_state(state, frame)

    def predict_from_state(
        self,
        state: GameState,
        frame: Optional[np.ndarray] = None,
    ) -> ShotParameters:
        """
        Predict shot parameters from a pre-computed GameState and optional
        raw frame.

        Parameters:
            state: GameState as produced by ScreenAnalyzer.
            frame: Optional raw game-screen frame (H×W×3 BGR).  If omitted a
                blank (zero) image is fed to the CNN branch, which means the
                prediction relies entirely on the feature vector.

        Returns:
            ShotParameters with direction, power, and loft recommendations.
        """
        feat_vec = torch.tensor(
            state.to_feature_vector(), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        if frame is not None:
            img_tensor = self._preprocess_frame(frame)
        else:
            img_tensor = torch.zeros(
                1, 3, self._img_h, self._img_w,
                dtype=torch.float32,
                device=self.device,
            )

        direction, power, loft = self._model.predict(img_tensor, feat_vec)
        return ShotParameters(
            direction_deg=direction,
            power=power,
            loft_deg=loft,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Resize and normalise a BGR frame into a ``(1, 3, H, W)`` float tensor
        on the target device with values in [0, 1].
        """
        import cv2  # imported here to avoid hard dep at module level

        resized = cv2.resize(
            frame, (self._img_w, self._img_h), interpolation=cv2.INTER_LINEAR
        )
        # Convert BGR → RGB and normalise to [0, 1]
        rgb = resized[:, :, ::-1].copy().astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
