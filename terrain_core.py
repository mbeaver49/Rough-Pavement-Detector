import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from glcm import (
    fast_glcm_contrast,
    fast_glcm_entropy,
    fast_glcm_ASM,
    fast_glcm_homogeneity,
    fast_glcm_correlation,
)


@dataclass
class GlcmFeatures:
    mean_contrast: float = 0.0
    mean_entropy: float = 0.0
    mean_energy: float = 0.0
    std_contrast: float = 0.0
    mean_homogeneity: float = 0.0
    mean_correlation: float = 0.0


@dataclass
class SafetyThresholds:
    max_contrast: float = 50.0
    max_entropy: float = 3.5
    max_std_contrast: float = 30.0
    min_homogeneity: float = 0.0
    min_correlation: float = 0.0
    min_energy: float = 0.0


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "unknown"
    is_safe: Optional[bool] = None
    features: Optional[GlcmFeatures] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def color(self) -> Tuple[int, int, int]:
        if self.is_safe is None:
            return (200, 200, 0)
        return (0, 200, 0) if self.is_safe else (0, 0, 220)

    @property
    def status_text(self) -> str:
        if self.is_safe is None:
            return "?"
        return "SAFE" if self.is_safe else "UNSAFE"


def extract_roi_features(
    gray: np.ndarray, x1: int, y1: int, x2: int, y2: int, levels: int = 6, ks: int = 21
) -> GlcmFeatures:
    """Crop the ROI from the grayscale image and compute GLCM statistics."""
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return GlcmFeatures()

    contrast = fast_glcm_contrast(patch, levels=levels, ks=ks)
    entropy = fast_glcm_entropy(patch, levels=levels, ks=ks)
    _, energy = fast_glcm_ASM(patch, levels=levels, ks=ks)
    homogeneity = fast_glcm_homogeneity(patch, levels=levels, ks=ks)
    correlation = fast_glcm_correlation(patch, levels=levels, ks=ks)

    return GlcmFeatures(
        mean_contrast=float(np.mean(contrast)),
        mean_entropy=float(np.mean(entropy)),
        mean_energy=float(np.mean(energy)),
        std_contrast=float(np.std(contrast)),
        mean_homogeneity=float(np.mean(homogeneity)),
        mean_correlation=float(np.mean(correlation)),
    )


def classify_roi(
    features: GlcmFeatures, thresholds: SafetyThresholds
) -> Tuple[bool, List[str]]:
    """Returns (is_safe, list_of_violated_reasons)."""
    reasons = []
    if features.mean_contrast > thresholds.max_contrast:
        reasons.append(
            f"contrast={features.mean_contrast:.1f}>{thresholds.max_contrast}"
        )
    if features.mean_entropy > thresholds.max_entropy:
        reasons.append(f"entropy={features.mean_entropy:.2f}>{thresholds.max_entropy}")
    if features.std_contrast > thresholds.max_std_contrast:
        reasons.append(
            f"std_contrast={features.std_contrast:.1f}>{thresholds.max_std_contrast}"
        )
    if features.mean_homogeneity < thresholds.min_homogeneity:
        reasons.append(
            f"homogeneity={features.mean_homogeneity:.2f}<{thresholds.min_homogeneity}"
        )
    if features.mean_correlation < thresholds.min_correlation:
        reasons.append(
            f"correlation={features.mean_correlation:.2f}<{thresholds.min_correlation}"
        )
    if features.mean_energy < thresholds.min_energy:
        reasons.append(f"energy={features.mean_energy:.2f}<{thresholds.min_energy}")
    return (len(reasons) == 0), reasons
