"""Lightweight, offline leaf segmentation and background normalization for PakAgriVision.

Isolates foreground leaves from field images (hands, soil, background weeds)
and composites them onto a neutral white canvas matching the PlantVillage
training distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "u2netp.onnx"


class LeafSegmenter:
    """Pre-processes field leaf photos to isolate the foliage on a white background."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.session: Optional[ort.InferenceSession] = None
        self.ready = False
        self._init_session()

    def _init_session(self) -> None:
        if ort is None:
            logging.warning("onnxruntime not installed; LeafSegmenter disabled.")
            return
        if not self.model_path.is_file():
            logging.warning("Leaf segmentation model not found at %s", self.model_path)
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.ready = True
            logging.info("LeafSegmenter initialized with %s", self.model_path.name)
        except Exception as exc:
            logging.warning("Failed to initialize LeafSegmenter: %s", exc)
            self.ready = False

    def segment_leaf(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Segment the leaf from the background and composite onto a neutral background.

        Args:
            image: Input PIL Image (RGB).
            threshold: Minimum saliency confidence (0.0 to 1.0) to treat as foreground.
            bg_color: RGB tuple for the background (default white: (255, 255, 255)).

        Returns:
            Tuple of (composited_image, mask_image).
            If the segmenter is not ready or fails, returns (original_image, None).
        """
        if not self.ready or self.session is None:
            return image, None

        orig_w, orig_h = image.size
        # 1. Resize to model input shape (320 x 320)
        resized = image.resize((320, 320), Image.BILINEAR)

        # 2. Normalize with standard ImageNet statistics
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)

        # 3. ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        pred = outputs[0][0, 0]

        # 4. Min-max normalize saliency map
        pred_min = float(pred.min())
        pred_max = float(pred.max())
        pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

        # 5. Soft alpha thresholding
        mask_data = (pred * 255.0).clip(0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_data, mode="L").resize((orig_w, orig_h), Image.BILINEAR)

        # 6. Composite onto specified background color
        canvas = Image.new("RGB", (orig_w, orig_h), bg_color)
        canvas.paste(image, mask=mask)

        return canvas, mask


# Shared singleton instance
SEGMENTER = LeafSegmenter()


def extract_leaf_on_white(image: Image.Image) -> Image.Image:
    """Convenience helper to extract leaf onto pure white background."""
    composited, _ = SEGMENTER.segment_leaf(image, bg_color=(255, 255, 255))
    return composited
