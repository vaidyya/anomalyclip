import base64
import io
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
# import pyrootutils  # Commented out - replaced with manual path setup
import torch
import torchvision.transforms as T
from PIL import Image

# Ensure project root on sys.path for `src.*` imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Limit CPU threads to avoid overloading system
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
except Exception:
    pass

from src.models.components.anomaly_clip import AnomalyCLIP
from src.models.components.clip import clip as clip_mod
from src.server.ollama_client import summarize_event

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    ckpt_path: str
    labels_file: str
    arch: str = "ViT-B/16"
    normal_id: int = 0
    # UCF-Crime defaults (EXACT match to video_dataset.py evaluation config)
    num_segments: int = int(os.getenv("SERVE_NUM_SEGMENTS", "32"))
    seg_length: int = int(os.getenv("SERVE_SEG_LEN", "16"))
    stride: int = int(os.getenv("SERVE_STRIDE", "1"))  # Original uses stride=1 for consecutive frames
    concat_features: bool = True
    emb_size: int = 256
    depth: int = 1
    heads: int = 4
    dim_heads: int = 64
    ncrops: int = 1
    select_idx_dropout_topk: float = 0.0
    select_idx_dropout_bottomk: float = 0.0
    num_topk: int = 3
    num_bottomk: int = 3
    ncentroid_path: Optional[str] = None


class ModelWrapper:
    """Wrapper for AnomalyCLIP model with automatic device selection and checkpoint inference."""
    
    def __init__(self, cfg: ServeConfig) -> None:
        self.cfg = cfg
        # Prefer CPU by default; allow override via ANOMALYCLIP_DEVICE={cpu,mps,cuda}
        dev_pref = os.getenv("ANOMALYCLIP_DEVICE", "cpu").lower()
        if dev_pref == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device (per env override)")
        elif dev_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device (per env override)")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        # Read labels first (used for compatibility inference)
        import pandas as pd
        try:
            labels_df = pd.read_csv(cfg.labels_file)
            self.class_ids: List[int] = labels_df["id"].tolist()
            self.class_names: List[str] = labels_df["name"].tolist()
            logger.info(f"Loaded {len(self.class_names)} class labels from {cfg.labels_file}")
        except Exception as e:
            logger.error(f"Failed to load labels from {cfg.labels_file}: {e}")
            raise

        # Inspect checkpoint to auto-match temporal model dims
        logger.info(f"Loading checkpoint from: {cfg.ckpt_path}")
        try:
            checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            logger.info(f"Checkpoint loaded, state_dict has {len(state_dict)} keys")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        # Find projection weight for emb_size/in_features inference
        projection_weight = None
        for key in [
            "net.temporal_model.projection.weight",
            "temporal_model.projection.weight",
        ]:
            if key in state_dict:
                projection_weight = state_dict[key]
                logger.info(f"Found projection weight at key: {key}")
                break
        
        if projection_weight is not None:
            inferred_emb_size = int(projection_weight.shape[0])  # projection out dim == emb_size
            cfg.emb_size = inferred_emb_size
            projection_input_dim = int(projection_weight.shape[1])
            # Common CLIP dims are 512 (ViT-B/16) or 768 (ViT-L/14)
            cfg.concat_features = projection_input_dim not in (512, 768)
            cfg.depth = 1  # Most released ckpts use depth=1
            logger.info(f"Inferred emb_size={cfg.emb_size}, concat_features={cfg.concat_features}")
        else:
            # Sensible defaults for released checkpoints
            cfg.emb_size = 128
            cfg.concat_features = False
            cfg.depth = 1
            logger.warning("Could not infer projection dims, using defaults")

        # Infer attention total dim (heads * dim_heads) from to_q/to_kv
        attention_total_dim = None
        for key in [
            "net.temporal_model.axial_attn.layers.blocks.0.f.net.fn.fn.to_q.weight",
            "temporal_model.axial_attn.layers.blocks.0.f.net.fn.fn.to_q.weight",
            "net.temporal_model.axial_attn.layers.blocks.0.g.net.fn.fn.to_q.weight",
            "temporal_model.axial_attn.layers.blocks.0.g.net.fn.fn.to_q.weight",
            "net.temporal_model.axial_attn.layers.blocks.0.f.net.fn.fn.to_kv.weight",
            "temporal_model.axial_attn.layers.blocks.0.f.net.fn.fn.to_kv.weight",
        ]:
            if key in state_dict:
                weight_tensor = state_dict[key]
                # to_q: [heads*dim_heads, emb_size], to_kv: [2*heads*dim_heads, emb_size]
                if weight_tensor.shape[0] % 2 == 0 and "to_kv" in key:
                    attention_total_dim = int(weight_tensor.shape[0] // 2)
                else:
                    attention_total_dim = int(weight_tensor.shape[0])
                logger.info(f"Inferred attention dim from {key}: {attention_total_dim}")
                break

        if attention_total_dim is not None:
            # Choose heads/dim_heads that exactly match attention_total_dim
            # Prefer keeping requested heads if it divides attention_total_dim
            if attention_total_dim % cfg.heads == 0:
                cfg.dim_heads = attention_total_dim // cfg.heads
            else:
                # Try common head counts
                for num_heads in (8, 4, 2, 1, 16):
                    if attention_total_dim % num_heads == 0:
                        cfg.heads = num_heads
                        cfg.dim_heads = attention_total_dim // num_heads
                        break
                else:
                    cfg.heads = 1
                    cfg.dim_heads = attention_total_dim
            logger.info(f"Set heads={cfg.heads}, dim_heads={cfg.dim_heads}")
        
        # Ensure heads*dim_heads is consistent with emb_size when possible
        if (cfg.heads * cfg.dim_heads) != attention_total_dim and attention_total_dim is not None:
            # Align to attention_total_dim inferred from checkpoint
            for num_heads in (cfg.heads, 8, 4, 2, 1, 16):
                if attention_total_dim % num_heads == 0:
                    cfg.heads = num_heads
                    cfg.dim_heads = attention_total_dim // num_heads
                    logger.info(f"Adjusted to heads={cfg.heads}, dim_heads={cfg.dim_heads}")
                    break

        # Infer depth (number of blocks)
        max_block_index = -1
        for key in state_dict.keys():
            if "temporal_model.axial_attn.layers.blocks" in key:
                try:
                    block_idx = int(key.split("layers.blocks.")[1].split(".")[0])
                    if block_idx > max_block_index:
                        max_block_index = block_idx
                except Exception as e:
                    logger.debug(f"Could not parse block index from key {key}: {e}")
        
        if max_block_index >= 0:
            cfg.depth = max_block_index + 1
            logger.info(f"Inferred depth={cfg.depth} from checkpoint")
        else:
            cfg.depth = cfg.depth or 1
            logger.info(f"Using default depth={cfg.depth}")

        # Build model with inferred-compatible hyperparams
        logger.info(f"Building AnomalyCLIP model with arch={cfg.arch}, emb_size={cfg.emb_size}, depth={cfg.depth}")
        self.net = AnomalyCLIP(
            arch=cfg.arch,
            labels_file=cfg.labels_file,
            emb_size=cfg.emb_size,
            depth=cfg.depth,
            heads=cfg.heads,
            dim_heads=cfg.dim_heads,
            num_segments=cfg.num_segments,
            seg_length=cfg.seg_length,
            concat_features=cfg.concat_features,
            normal_id=cfg.normal_id,
            stride=cfg.stride,
            load_from_features=False,
            select_idx_dropout_topk=cfg.select_idx_dropout_topk,
            select_idx_dropout_bottomk=cfg.select_idx_dropout_bottomk,
            ncrops=cfg.ncrops,
            num_topk=cfg.num_topk,
            num_bottomk=cfg.num_bottomk,
            # Prompt learner defaults for inference
            n_ctx=8,
            shared_context=False,
            ctx_init="",
        )
        self.net.eval().to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Load weights from lightning checkpoint (filter to net.*), but skip shape-mismatched keys
        raw_net_state = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
        current_state = self.net.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in raw_net_state.items():
            if k in current_state and tuple(v.shape) == tuple(current_state[k].shape):
                filtered_state[k] = v
            else:
                skipped_keys.append(k)
        
        missing_keys, unexpected_keys = self.net.load_state_dict(filtered_state, strict=False)
        if skipped_keys:
            logger.warning(f"Skipped {len(skipped_keys)} mismatched keys (e.g., {skipped_keys[:3]})")
        if missing_keys:
            logger.warning(f"Missing keys when loading state: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state: {unexpected_keys[:5]}...")
        logger.info("Model weights loaded (with filtered keys)")

        # Preprocess using CLIP's own preprocessing for the chosen arch
        logger.info(f"Loading CLIP preprocessing for {cfg.arch}")
        _, self.preprocess = clip_mod.load(cfg.arch, device="cpu")

        # Insert normal class name info
        if len(self.class_names) == len(self.class_ids):
            self.num_classes = len(self.class_names) + 1  # +1 for normal
        else:
            self.num_classes = len(self.class_names)

        # Normal centroid for anomaly detection
        self.ncentroid: Optional[torch.Tensor] = None
        if cfg.ncentroid_path and Path(cfg.ncentroid_path).exists():
            logger.info(f"Loading normal centroid from: {cfg.ncentroid_path}")
            self.ncentroid = torch.load(cfg.ncentroid_path, map_location=self.device)
        else:
            logger.info("No normal centroid provided, will compute from video frames")

    def ensure_ncentroid_from_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Ensure normal centroid is computed, using provided frames if necessary.
        
        Args:
            frames: List of BGR numpy arrays to use for centroid computation
            
        Returns:
            Normal centroid tensor
        """
        if self.ncentroid is not None:
            return self.ncentroid.to(self.device)
        
        # Approximate using first K frames assumed mostly normal
        logger.info(f"Computing normal centroid from {len(frames)} frames")
        preprocessed_images = [self.preprocess(Image.fromarray(frame[..., ::-1])) for frame in frames]
        image_batch = torch.stack(preprocessed_images, dim=0).to(self.device)  # [K,3,H,W]
        
        with torch.no_grad():
            frame_features = self.net.image_encoder(image_batch)
            self.ncentroid = frame_features.mean(dim=0)
        
        logger.info("Normal centroid computed successfully")
        return self.ncentroid

    def window_infer(
        self,
        window_frames: List[np.ndarray],
        labels_stub: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a window of frames.
        
        Args:
            window_frames: List of BGR numpy arrays (must be num_segments * seg_length)
            labels_stub: Optional label tensor
            
        Returns:
            Tuple of (similarity scores, anomaly scores)
        """
        expected_num_frames = self.cfg.num_segments * self.cfg.seg_length
        if len(window_frames) != expected_num_frames:
            raise ValueError(f"Expected {expected_num_frames} frames, got {len(window_frames)}")
        
        # Preprocess each frame to CLIP input format
        pil_frames = [Image.fromarray(frame[..., ::-1]) for frame in window_frames]  # BGR->RGB
        preprocessed_frames = [self.preprocess(pil_img) for pil_img in pil_frames]  # list of CHW tensors
        frame_tensor = torch.stack(preprocessed_frames, dim=0)  # [t,3,H,W]
        frame_tensor = frame_tensor.unsqueeze(0)  # [1,t,3,H,W]
        
        labels = labels_stub if labels_stub is not None else torch.zeros(expected_num_frames, dtype=torch.long)
        normal_centroid = self.ncentroid.to(self.device)

        with torch.no_grad():
            similarity_scores, anomaly_scores = self.net(
                frame_tensor.to(self.device),
                labels.to(self.device),
                normal_centroid,
                segment_size=1,
                test_mode=True,
            )
        
        return similarity_scores.detach().cpu(), anomaly_scores.detach().cpu()


def jpeg_b64(img_bgr: np.ndarray, quality: int = 80) -> str:
    """Encode BGR image to base64 JPEG string.
    
    Args:
        img_bgr: BGR numpy array
        quality: JPEG quality (1-100)
        
    Returns:
        Base64-encoded JPEG string
    """
    encode_success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not encode_success:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def draw_overlay(img: np.ndarray, score: float, topk: List[Tuple[str, float]]) -> np.ndarray:
    """Draw anomaly score overlay on image.
    
    Args:
        img: BGR image
        score: Anomaly score (0-1)
        topk: List of (class_name, probability) tuples
        
    Returns:
        Image with overlay drawn
    """
    height, width = img.shape[:2]
    output_image = img.copy()
    
    # Draw border (red if anomaly, blue if normal)
    border_color = (0, 0, 255) if score >= 0.5 else (255, 0, 0)
    cv2.rectangle(output_image, (0, 0), (width - 1, height - 1), border_color, 4)

    # Draw translucent panel on right side
    panel_width = int(width * 0.35)
    panel = np.full((height, panel_width, 3), (16, 16, 16), dtype=np.uint8)
    alpha = 0.65
    output_image[:, width - panel_width : width] = cv2.addWeighted(
        output_image[:, width - panel_width : width], 1 - alpha, panel, alpha, 0
    )

    # Draw text and bars
    text_y_start = 40
    text_x_start = width - panel_width + 20
    cv2.putText(
        output_image, f"Anomaly: {score:.2f}", (text_x_start, text_y_start),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA
    )
    
    current_y = text_y_start + 30
    for class_label, probability in topk[:5]:
        bar_width = int((panel_width - 60) * max(min(probability, 1.0), 0.0))
        cv2.putText(
            output_image, class_label[:18], (text_x_start, current_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.rectangle(
            output_image, (text_x_start, current_y + 28),
            (text_x_start + bar_width, current_y + 48), (0, 140, 255), -1
        )
        cv2.putText(
            output_image, f"{probability:.2f}", (text_x_start + bar_width + 8, current_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
        )
        current_y += 40
    
    return output_image


def run_realtime(
    model: ModelWrapper,
    video_path: str,
    send: Callable[[Dict[str, Any]], None],
    alert_threshold: float = 0.5,
    alert_min_frames: int = 8,
    infer_interval: int = 4,
    max_emit_fps: float = 8.0,
) -> None:
    """Run real-time anomaly detection on video with streaming results.
    
    Args:
        model: Initialized ModelWrapper
        video_path: Path to input video file
        send: Callback function to send results
        alert_threshold: Score threshold for anomaly alerts
        alert_min_frames: Minimum consecutive frames for alert
        infer_interval: Run inference every N frames
        max_emit_fps: Maximum rate to send results
    """
    logger.info(f"Starting real-time inference on: {video_path}")
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            error_msg = f"Cannot open video file: {video_path}"
            logger.error(error_msg)
            send({"error": error_msg})
            return

        video_fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
        if not video_fps or video_fps != video_fps:  # Check for NaN
            video_fps = 25.0
            logger.warning(f"Could not detect video FPS, using default: {video_fps}")
        else:
            logger.info(f"Video FPS: {video_fps}")

        window_size = model.cfg.num_segments * model.cfg.seg_length
        frame_buffer: List[np.ndarray] = []

        # Bootstrap normal centroid with first 64 frames or up to window_size
        warmup_frames: List[np.ndarray] = []
        max_width = int(os.getenv("SERVE_MAX_WIDTH", "0"))
        while len(warmup_frames) < min(64, window_size):
            read_success, frame = video_capture.read()
            if not read_success:
                break
            if frame is not None and max_width > 0 and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            warmup_frames.append(frame)
        
        if not warmup_frames:
            video_capture.release()
            error_msg = "Video is empty or unreadable"
            logger.error(error_msg)
            send({"error": error_msg})
            return
        
        logger.info(f"Computing normal centroid from {len(warmup_frames)} warmup frames")
        model.ensure_ncentroid_from_frames(warmup_frames)

        # Reset to first frame (process from beginning)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info("Starting frame-by-frame processing")

        frame_index = 0
        consecutive_anomaly_frames = 0
        reference_alert_image_b64: Optional[str] = None
        last_anomaly_scores = None
        last_class_probabilities_full = None
        last_emit_time = 0.0
        warmup_start_time = time.time()
        # Throttle settings
        env_infer_interval = os.getenv("SERVE_INFER_INTERVAL")
        if env_infer_interval:
            try:
                infer_interval = int(env_infer_interval)
            except Exception:
                pass
        env_max_emit = os.getenv("SERVE_MAX_EMIT_FPS")
        if env_max_emit:
            try:
                max_emit_fps = float(env_max_emit)
            except Exception:
                pass
        max_width = int(os.getenv("SERVE_MAX_WIDTH", "0"))

        # First pass: collect all frames into memory to determine total video length
        all_frames = []
        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break
            # Downscale to limit memory/compute
            if current_frame is not None and max_width > 0 and current_frame.shape[1] > max_width:
                scale = max_width / current_frame.shape[1]
                current_frame = cv2.resize(
                    current_frame,
                    (int(current_frame.shape[1] * scale), int(current_frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            all_frames.append(current_frame)
        
        total_frames = len(all_frames)
        logger.info(f"Video loaded: {total_frames} total frames")
        
        if total_frames == 0:
            video_capture.release()
            send({"error": "Video contains no frames"})
            return
        
        # Decide processing strategy based on video length
        if total_frames >= window_size:
            # LONG VIDEO: Use sliding window approach (matching original test_mode evaluation)
            logger.info(f"Long video mode: processing {total_frames} frames with {window_size}-frame sliding window")
            
            # Calculate stride for sliding window (how much to shift between windows)
            # Use smaller stride for overlapping windows to not miss anomalies
            slide_stride = window_size // 4  # 25% overlap
            num_windows = max(1, (total_frames - window_size) // slide_stride + 1)
            logger.info(f"Will process {num_windows} overlapping windows (stride={slide_stride})")
            
            # Process each window and accumulate results
            all_frame_scores = {}  # frame_idx -> list of scores from different windows
            all_frame_probs = {}   # frame_idx -> list of prob vectors
            normal_class_index = model.cfg.normal_id
            
            for window_idx in range(num_windows):
                window_start = window_idx * slide_stride
                window_end = min(window_start + window_size, total_frames)
                
                # Extract window frames
                window_frames = all_frames[window_start:window_end]
                
                # Pad if last window is shorter
                if len(window_frames) < window_size:
                    padding = [window_frames[-1]] * (window_size - len(window_frames))
                    window_frames = window_frames + padding
                
                logger.info(f"Processing window {window_idx + 1}/{num_windows}: frames {window_start}-{window_end-1}")
                
                # Run inference on this window
                similarity_matrix, anomaly_scores = model.window_infer(window_frames)
                
                # Expand per-frame class probabilities
                softmax_similarity = torch.softmax(similarity_matrix, dim=1)
                anomaly_class_probs = softmax_similarity * anomaly_scores.unsqueeze(1)
                normal_class_probs = 1 - anomaly_scores
                
                left_probs = anomaly_class_probs[:, :normal_class_index]
                right_probs = anomaly_class_probs[:, normal_class_index:]
                full_class_probs = torch.cat(
                    [left_probs, normal_class_probs.unsqueeze(1), right_probs], dim=1
                )
                
                # Store results for each frame in this window
                actual_frames_in_window = min(window_size, window_end - window_start)
                for i in range(actual_frames_in_window):
                    global_frame_idx = window_start + i
                    score = float(anomaly_scores[i].item())
                    probs = full_class_probs[i].numpy().tolist()
                    
                    if global_frame_idx not in all_frame_scores:
                        all_frame_scores[global_frame_idx] = []
                        all_frame_probs[global_frame_idx] = []
                    
                    all_frame_scores[global_frame_idx].append(score)
                    all_frame_probs[global_frame_idx].append(probs)
            
            # Average scores from overlapping windows
            logger.info("Averaging scores from overlapping windows...")
            final_scores = {}
            final_probs = {}
            for frame_idx in range(total_frames):
                if frame_idx in all_frame_scores:
                    # Average all predictions for this frame
                    final_scores[frame_idx] = np.mean(all_frame_scores[frame_idx])
                    final_probs[frame_idx] = np.mean(all_frame_probs[frame_idx], axis=0).tolist()
                else:
                    # Should not happen, but fallback to 0
                    final_scores[frame_idx] = 0.0
                    final_probs[frame_idx] = [1.0 / (len(model.class_names) + 1)] * (len(model.class_names) + 1)
            
            # Log statistics
            scores_array = np.array([final_scores[i] for i in range(total_frames)])
            logger.info(f"Final anomaly scores - Min: {scores_array.min():.3f}, Max: {scores_array.max():.3f}, Mean: {scores_array.mean():.3f}")
            logger.info(f"Scores >= 0.5: {(scores_array >= 0.5).sum()}/{len(scores_array)} frames")
            
            # Stream results for ALL frames (emit every Nth frame to limit bandwidth)
            emit_interval = max(1, int(total_frames / 500))  # Target ~500 frames max
            logger.info(f"Emitting every {emit_interval} frame(s) (total {total_frames // emit_interval} frames will be sent)")
            
            for frame_idx in range(total_frames):
                current_anomaly_score = final_scores[frame_idx]
                current_class_probs = final_probs[frame_idx]
                
                # Only emit every Nth frame to reduce bandwidth
                if frame_idx % emit_interval != 0:
                    continue
                
                # Get ALL ANOMALY classes (excluding normal), sorted by probability
                anomaly_only_probs = [
                    (prob, idx) for idx, prob in enumerate(current_class_probs)
                    if idx != normal_class_index
                ]
                anomaly_only_probs.sort(reverse=True, key=lambda x: x[0])
                topk_class_indices = [idx for _, idx in anomaly_only_probs]
                
                # Build class pairs for display
                topk_class_pairs = [
                    (
                        model.class_names[k - (0 if k < normal_class_index else 1)] if k != normal_class_index else "normal",
                        float(current_class_probs[k]),
                    )
                    for k in topk_class_indices
                ]
                
                # Use original frame for overlay
                overlay_image = draw_overlay(all_frames[frame_idx], current_anomaly_score, topk_class_pairs)
                
                payload = {
                    "frame_index": frame_idx + 1,  # 1-based indexing
                    "abnormal_score": current_anomaly_score,
                    "topk": [
                        {"label": name, "prob": prob, "class_index": int(k)}
                        for (name, prob), k in zip(topk_class_pairs, topk_class_indices)
                    ],
                    "image_b64_jpeg": jpeg_b64(overlay_image, quality=70),
                    "alert": None,  # Simplified for long videos
                    "meta": {"fps": video_fps}
                }
                
                send(payload)
                logger.debug(f"Sent frame {frame_idx + 1}/{total_frames}: score={current_anomaly_score:.3f}")
        else:
            # SHORT VIDEO: Use padding approach (existing logic)
            logger.info(f"Short video detected ({total_frames} frames), padding to {window_size}")
            padding_frames = [all_frames[-1]] * (window_size - total_frames)
            padded_window = all_frames + padding_frames

            similarity_matrix, anomaly_scores = model.window_infer(padded_window)
            softmax_similarity = torch.softmax(similarity_matrix, dim=1)
            anomaly_class_probs = softmax_similarity * anomaly_scores.unsqueeze(1)
            normal_class_probs = 1 - anomaly_scores
            normal_class_index = model.cfg.normal_id

            left_probs = anomaly_class_probs[:, :normal_class_index]
            right_probs = anomaly_class_probs[:, normal_class_index:]
            full_class_probs = torch.cat([left_probs, normal_class_probs.unsqueeze(1), right_probs], dim=1)

            # Align last total_frames entries with original frames
            start_idx = window_size - total_frames
            for i in range(total_frames):
                idx = start_idx + i
                score_i = float(anomaly_scores[idx].item())
                probs_i = full_class_probs[idx].numpy().tolist()
                
                # Get ALL ANOMALY classes (excluding normal)
                anomaly_only_probs_short = [
                    (prob, k) for k, prob in enumerate(probs_i)
                    if k != normal_class_index
                ]
                anomaly_only_probs_short.sort(reverse=True, key=lambda x: x[0])
                topk_idx = [k for _, k in anomaly_only_probs_short]
                
                topk_pairs = [
                    (
                        model.class_names[k - (0 if k < normal_class_index else 1)] if k != normal_class_index else "normal",
                        float(probs_i[k]),
                    )
                    for k in topk_idx
                ]
                overlay_img = draw_overlay(all_frames[i], score_i, topk_pairs)

                payload = {
                    "frame_index": i + 1,
                    "abnormal_score": score_i,
                    "topk": [
                        {"label": name, "prob": prob, "class_index": int(k)}
                        for (name, prob), k in zip(topk_pairs, topk_idx)
                    ],
                    "image_b64_jpeg": jpeg_b64(overlay_img, quality=70),
                    "alert": None,
                    "meta": {"fps": video_fps},
                }
                logger.debug(f"Sending frame {i+1}/{total_frames}: score={score_i:.3f}")
                send(payload)
            
            # For short videos, collect scores for summary
            final_scores = {i: float(anomaly_scores[start_idx + i].item()) for i in range(total_frames)}
        
        # Generate video summary using Ollama after processing complete
        logger.info("Generating video summary with Ollama...")

        # Build a data-driven fallback summary to always send something
        def ts_from_frame(fi: int) -> str:
            secs = (fi / video_fps) if video_fps else 0.0
            m = int(secs // 60)
            s = int(secs % 60)
            return f"{m:02d}:{s:02d}"

        scores_list = [float(final_scores.get(i, 0)) for i in range(total_frames)] if 'final_scores' in locals() else [0.0] * total_frames
        total_anomalies = int(sum(1 for s in scores_list if s >= 0.5))
        rate = (total_anomalies / total_frames * 100.0) if total_frames else 0.0
        if scores_list:
            peak_i = int(np.argmax(scores_list))
            peak_score = float(scores_list[peak_i])
        else:
            peak_i = 0
            peak_score = 0.0
        fallback_summary = {
            "overall_summary": f"Data summary: {total_anomalies} anomalous frames ({rate:.1f}%). Peak {peak_score:.2f} at frame {peak_i} ({ts_from_frame(peak_i)}).",
            "anomaly_segments": [],
            "statistics": {
                "total_frames": int(total_frames),
                "total_anomalies": int(total_anomalies),
                "anomaly_rate_percent": round(rate, 2),
                "peak_anomaly": {
                    "score": peak_score,
                    "frame": peak_i,
                    "timestamp": ts_from_frame(peak_i),
                },
            },
            "keyframes": [],
        }

        try:
            from src.server.ollama_client import generate_video_summary
            import asyncio
            import base64
            
            # Collect all frame results for summary
            frame_results = []
            for i in range(total_frames):
                frame_results.append({
                    'frame_index': i,
                    'abnormal_score': final_scores.get(i, 0) if 'final_scores' in locals() else 0
                })
            
            video_metadata = {
                'duration': total_frames / video_fps,
                'fps': video_fps,
                'total_frames': total_frames
            }
            
            # Select keyframes and capture frame images for VLM
            # Notify frontend that summary generation is starting
            send({'type': 'status', 'message': 'Generating AI video summary...'})
            logger.info("Starting AI video summary generation using statistical analysis")

            # Generate summary using statistical analysis (no image processing needed)
            summary_result = asyncio.run(generate_video_summary(
                frame_results,
                video_metadata
            ))

            # If the model returned an empty or invalid summary, fallback
            if not isinstance(summary_result, dict) or not summary_result.get("overall_summary"):
                logger.warning("Ollama summary empty/invalid; sending fallback summary")
                send({'type': 'summary', 'summary': fallback_summary})
            else:
                # Send summary to frontend
                send({'type': 'summary', 'summary': summary_result})
                logger.info("Video summary sent to frontend")
                send({'type': 'status', 'message': 'AI summary complete'})
        except Exception as e:
            logger.error(f"Failed to generate video summary: {e}")
            # Always send fallback on failure
            send({'type': 'summary', 'summary': fallback_summary})
            send({'type': 'status', 'message': 'Basic analysis complete (AI summary unavailable)'})

        video_capture.release()
        logger.info("Video processing complete")
    except Exception as e:
        logger.error(f"Error during real-time inference: {e}", exc_info=True)
        try:
            send({"error": f"Inference error: {str(e)}"})
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


def analyze_offline(
    model: ModelWrapper,
    video_path: str,
    step: int = 1,
) -> Dict[str, Any]:
    """Run offline analysis on entire video and return all results.
    
    Args:
        model: Initialized ModelWrapper
        video_path: Path to video file
        step: Process every Nth frame (1 = process all frames)
        
    Returns:
        Dictionary with fps, frame indices, scores, and top-k results
    """
    logger.info(f"Starting offline analysis of: {video_path}")
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
    window_size = model.cfg.num_segments * model.cfg.seg_length
    logger.info(f"Video FPS: {video_fps}, window size: {window_size}")

    # Warmup centroid computation
    warmup_frames: List[np.ndarray] = []
    while len(warmup_frames) < min(64, window_size):
        read_success, frame = video_capture.read()
        if not read_success:
            break
        warmup_frames.append(frame)
    
    if not warmup_frames:
        raise RuntimeError("Video is empty or unreadable")
    
    logger.info(f"Computing normal centroid from {len(warmup_frames)} frames")
    model.ensure_ncentroid_from_frames(warmup_frames)

    # Rewind to beginning
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_buffer: List[np.ndarray] = []
    frame_index = 0
    result_indices: List[int] = []
    result_scores: List[float] = []
    result_topk: List[List[Tuple[str, float]]] = []

    normal_class_index = model.cfg.normal_id

    while True:
        read_success, frame = video_capture.read()
        if not read_success:
            break
        
        frame_buffer.append(frame)
        frame_index += 1

        if len(frame_buffer) < window_size:
            continue
        
        if len(frame_buffer) > window_size:
            frame_buffer = frame_buffer[-window_size:]

        # Process every Nth frame
        if (frame_index % max(step, 1)) != 0:
            continue

        # Run inference
        similarity_matrix, anomaly_scores = model.window_infer(frame_buffer)
        softmax_similarity = torch.softmax(similarity_matrix, dim=1)
        anomaly_class_probs = softmax_similarity * anomaly_scores.unsqueeze(1)
        normal_class_probs = 1 - anomaly_scores
        
        left_probs = anomaly_class_probs[:, :normal_class_index]
        right_probs = anomaly_class_probs[:, normal_class_index:]
        full_class_probs = torch.cat([left_probs, normal_class_probs.unsqueeze(1), right_probs], dim=1)
        
        most_recent_idx = -1
        current_score = float(anomaly_scores[most_recent_idx].item())
        current_probs = full_class_probs[most_recent_idx].numpy().tolist()
        topk_indices = np.argsort(current_probs)[::-1][:5]
        
        topk_pairs = [
            (
                model.class_names[k - (0 if k < normal_class_index else 1)] if k != normal_class_index else "normal",
                float(current_probs[k]),
            )
            for k in topk_indices
        ]
        
        result_indices.append(frame_index)
        result_scores.append(current_score)
        result_topk.append(topk_pairs)

    video_capture.release()
    logger.info(f"Offline analysis complete: {len(result_indices)} frames processed")

    return {
        "fps": float(video_fps),
        "indices": result_indices,
        "scores": result_scores,
        "topk": result_topk,
    }
