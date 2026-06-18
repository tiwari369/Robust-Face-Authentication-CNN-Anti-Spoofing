"""
Lightweight CNN-based anti-spoofing module for LBAMS.

This file avoids heavy deep-learning dependencies so the project remains practical on
Python 3.13.x in VS Code. The module follows the same anti-spoofing idea described
in the conference paper: a convolutional texture analyser is used as a second
security signal along with blink-based liveness verification.

The detector is intentionally conservative by default: it produces a live/spoof
score and audit evidence, while the runtime configuration decides whether this
score should block attendance or only be displayed as an anti-spoofing audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class CNNAntiSpoofResult:
    score: float
    is_live: bool
    label: str
    evidence: Dict[str, float]


class LightweightCNNAntiSpoof:
    """Small convolutional anti-spoofing scorer implemented with OpenCV/NumPy.

    The model uses fixed convolution filters, ReLU activations, pooling, and a
    compact logistic head over texture, high-frequency, and convolutional energy
    statistics. It is not meant to replace a large trained neural network, but it
    gives the project a reproducible CNN-style anti-spoofing component without
    forcing TensorFlow/PyTorch installation on Python 3.13 laptops.
    """

    def __init__(self, threshold: float = 0.42):
        self.threshold = float(threshold)
        self.kernels = [
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32),
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        ]

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -12.0, 12.0))
        return 1.0 / (1.0 + float(np.exp(-x)))

    @staticmethod
    def _max_pool_2x2(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        if h2 <= 0 or w2 <= 0:
            return x
        return x[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))

    def _preprocess(self, face_gray: np.ndarray) -> np.ndarray:
        if face_gray.ndim == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face_gray, (96, 96))
        face = cv2.GaussianBlur(face, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face = clahe.apply(face)
        return face.astype(np.float32) / 255.0

    def predict(self, face_gray: np.ndarray) -> CNNAntiSpoofResult:
        face = self._preprocess(face_gray)

        conv_means = []
        conv_stds = []
        for kernel in self.kernels:
            response = cv2.filter2D(face, cv2.CV_32F, kernel)
            response = np.maximum(response, 0.0)  # ReLU
            pooled = self._max_pool_2x2(response)
            conv_means.append(float(np.mean(pooled)))
            conv_stds.append(float(np.std(pooled)))

        lap_var = float(cv2.Laplacian((face * 255).astype(np.uint8), cv2.CV_64F).var())
        grad_x = cv2.Sobel(face, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = float(np.mean(np.sqrt(grad_x * grad_x + grad_y * grad_y)))

        fft = np.fft.fftshift(np.fft.fft2(face))
        magnitude = np.log1p(np.abs(fft))
        center = magnitude[36:60, 36:60]
        high_frequency = float((magnitude.sum() - center.sum()) / (magnitude.sum() + 1e-7))

        brightness = float(np.mean(face))
        contrast = float(np.std(face))
        conv_energy = float(np.mean(conv_means) + np.mean(conv_stds))

        # Normalised evidence values. Real camera faces usually keep moderate
        # contrast, edge energy, and high-frequency skin/lighting details.
        tex = np.clip(lap_var / 85.0, 0.0, 1.7)
        edge = np.clip(grad_energy / 0.20, 0.0, 1.6)
        hf = np.clip((high_frequency - 0.55) / 0.25, 0.0, 1.3)
        cnn = np.clip(conv_energy / 0.65, 0.0, 1.6)
        exposure_penalty = 0.0
        if brightness < 0.18 or brightness > 0.82:
            exposure_penalty += 0.35
        if contrast < 0.08:
            exposure_penalty += 0.35

        logit = -1.25 + 0.65 * tex + 0.55 * edge + 0.45 * hf + 0.50 * cnn - exposure_penalty
        score = self._sigmoid(logit)
        is_live = bool(score >= self.threshold)
        label = "live-face-signal" if is_live else "spoof-risk-signal"
        evidence = {
            "laplacian_texture": lap_var,
            "gradient_energy": grad_energy,
            "high_frequency_ratio": high_frequency,
            "cnn_activation_energy": conv_energy,
            "brightness": brightness,
            "contrast": contrast,
            "threshold": self.threshold,
        }
        return CNNAntiSpoofResult(score=score, is_live=is_live, label=label, evidence=evidence)
