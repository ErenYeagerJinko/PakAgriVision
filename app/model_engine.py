"""Thread-safe Flask adapter for the shared inference and Grad-CAM modules."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image


class ModelUnavailable(RuntimeError):
    """A trained, class-compatible checkpoint is unavailable."""


class ModelEngine:
    """Load a trained model once; never return demo predictions on failure."""

    def __init__(self, checkpoint_path: Path, class_names: list[str]):
        self.device = "cpu"
        self.ready = False
        self.error = ""
        self._lock = RLock()
        self.class_names = class_names
        try:
            if not checkpoint_path.is_file():
                raise FileNotFoundError("Download runs/baseline/best.pt before diagnosing images")
            from src.inference import InferenceEngine
            from src.explainability.gradcam import generate_gradcam_visualization

            self._visualize = generate_gradcam_visualization
            self.inference = InferenceEngine(checkpoint_path=checkpoint_path)
            if self.inference.class_names != class_names:
                raise ValueError("Model and application class manifests disagree")
            self.device = str(self.inference.device)
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def _check_ready(self) -> None:
        if not self.ready:
            raise ModelUnavailable(self.error or "Trained model is unavailable")

    def predict(self, image: Image.Image, filter_crop: str | None = None) -> dict[str, Any]:
        self._check_ready()
        with self._lock:
            result = self.inference.predict(image, top_k=len(self.class_names))
        candidates = result["top_k"]
        if filter_crop and filter_crop.lower() != "all":
            # Rank within the selected crop, but keep original 38-class probabilities.
            # Renormalizing after masking would inflate confidence and bypass safeguards.
            candidates = [c for c in candidates if c["crop"].casefold() == filter_crop.casefold()]
            if not candidates:
                raise ValueError("Unknown crop filter")
        top3 = [
            {"class": c["class_name"], "crop": c["crop"], "disease": c["condition"],
             "confidence": c["probability"] * 100.0}
            for c in candidates[:3]
        ]
        best = top3[0]
        return {
            "predicted_class": best["class"], "crop": best["crop"],
            "disease": best["disease"], "confidence": best["confidence"],
            "is_healthy": best["class"].endswith("___healthy"),
            "top_3": top3, "latency_ms": result["latency_ms"],
        }

    def generate_gradcam(self, image: Image.Image, target_class: str | None = None,
                         alpha: float = 0.55) -> Image.Image:
        self._check_ready()
        if target_class is not None and target_class not in self.class_names:
            raise ValueError("Unknown Grad-CAM target class")
        with self._lock:
            from torchvision.transforms.functional import to_pil_image
            from src.data.transforms import denormalize_tensor

            tensor = self.inference.eval_transform(image.convert("RGB")).unsqueeze(0)
            # Render over the exact center-cropped model view, not a stretched full image.
            # Otherwise the heatmap would be spatially misregistered on non-square photos.
            model_view = to_pil_image(denormalize_tensor(tensor.squeeze(0)))
            _, overlay = self._visualize(
                self.inference.model, tensor, model_view,
                target_class_idx=self.class_names.index(target_class) if target_class else None,
                alpha=alpha,
            )
        return overlay
