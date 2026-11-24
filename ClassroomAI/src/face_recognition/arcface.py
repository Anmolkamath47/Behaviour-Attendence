"""
ClassroomAI - ArcFace 512-D Feature Embedder
Generates unit-norm 512-dimensional face embeddings using ArcFace backbone.
Optimized for CPU inference with ONNX Runtime and PyTorch fallback.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger("ClassroomAI.ArcFace")


class ConvBlock(nn.Module):
    """Standard Convolution-BatchNorm-PReLU block."""
    def __init__(self, in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    """Linear Convolution-BatchNorm block without activation (used before pooling)."""
    def __init__(self, in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthwiseSeparable(nn.Module):
    """Inverted Residual Bottleneck Block for MobileFaceNet ArcFace."""
    def __init__(self, in_c, out_c, stride=(1, 1), expansion=2):
        super().__init__()
        hidden_dim = in_c * expansion
        self.use_residual = (stride == (1, 1) or stride == 1) and in_c == out_c

        self.conv1 = ConvBlock(in_c, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.dw = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim)
        self.conv2 = LinearBlock(hidden_dim, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv2(self.dw(self.conv1(x)))
        if self.use_residual:
            return x + out
        return out


class ArcFaceMobileFaceNet(nn.Module):
    """
    MobileFaceNet ArcFace architecture producing 512-dimensional normalized embeddings.
    Input shape: (B, 3, 112, 112)
    Output shape: (B, 512)
    """
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.dw1 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, groups=64)

        # Inverted residual stages
        self.stage1 = nn.Sequential(
            DepthwiseSeparable(64, 64, stride=2, expansion=2),
            DepthwiseSeparable(64, 64, stride=1, expansion=2),
            DepthwiseSeparable(64, 64, stride=1, expansion=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparable(64, 128, stride=2, expansion=2),
            DepthwiseSeparable(128, 128, stride=1, expansion=2),
            DepthwiseSeparable(128, 128, stride=1, expansion=2),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(128, 128, stride=2, expansion=2),
            DepthwiseSeparable(128, 128, stride=1, expansion=2),
        )

        self.conv2 = ConvBlock(128, 512, kernel_size=1, stride=1, padding=0)
        # Global depthwise pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dw1(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.bn_out(self.linear(out))
        # L2-normalization for cosine distance
        return F.normalize(out, p=2, dim=1)


class ArcFaceEmbedder:
    """
    ArcFace Feature Extractor with dual runtime execution:
    - High-performance ONNX Runtime CPUExecutionProvider (when model exists or after export)
    - Native PyTorch CPU execution fallback
    """

    def __init__(self, model_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.onnx_path = self.project_root / "models" / "arcface" / "arcface.onnx"
        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)

        self.ort_session = None
        self.pytorch_model = None

        if model_path and os.path.exists(model_path) and str(model_path).endswith(".onnx"):
            self.onnx_path = Path(model_path)

        # Attempt to load or export ONNX model
        if ONNX_AVAILABLE and self.onnx_path.exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    opts,
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"Loaded ArcFace ONNX model: {self.onnx_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")

        if self.ort_session is None:
            self._init_pytorch_and_export_onnx()

    def _init_pytorch_and_export_onnx(self):
        """Initializes PyTorch ArcFace model and exports to ONNX for CPU acceleration."""
        self.pytorch_model = ArcFaceMobileFaceNet(embedding_size=512)
        self.pytorch_model.eval()
        logger.info("Initialized ArcFace MobileFaceNet in PyTorch.")

        if ONNX_AVAILABLE and not self.onnx_path.exists():
            try:
                logger.info(f"Exporting ArcFace to ONNX: {self.onnx_path}")
                dummy_input = torch.randn(1, 3, 112, 112)
                torch.onnx.export(
                    self.pytorch_model,
                    dummy_input,
                    str(self.onnx_path),
                    input_names=["input"],
                    output_names=["embedding"],
                    opset_version=18
                )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    opts,
                    providers=["CPUExecutionProvider"]
                )
                logger.info("ArcFace successfully exported and loaded in ONNX Runtime.")
            except Exception as e:
                logger.warning(f"Could not export ONNX model: {e}. Using PyTorch CPU.")

    def preprocess(self, face_rgb_112: np.ndarray) -> np.ndarray:
        """
        Normalize 112x112 RGB image to standard ArcFace input:
        (x - 127.5) / 128.0 with shape (1, 3, 112, 112)
        """
        assert face_rgb_112.shape[:2] == (112, 112), f"Expected (112, 112), got {face_rgb_112.shape[:2]}"
        img = face_rgb_112.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))  # (3, 112, 112)
        return np.expand_dims(img, axis=0)  # (1, 3, 112, 112)

    def extract_embedding(self, face_rgb_112: np.ndarray) -> np.ndarray:
        """
        Extract normalized 512-D embedding from a 112x112 RGB face crop.

        Returns:
            np.ndarray of shape (512,), L2-normalized.
        """
        inp = self.preprocess(face_rgb_112)

        if self.ort_session is not None:
            # ONNX Runtime inference
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: inp}
            ort_outs = self.ort_session.run(None, ort_inputs)
            emb = ort_outs[0][0]
        else:
            # PyTorch inference
            with torch.no_grad():
                tensor_in = torch.from_numpy(inp)
                emb = self.pytorch_model(tensor_in).cpu().numpy()[0]

        # Ensure unit norm
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb.astype(np.float32)
