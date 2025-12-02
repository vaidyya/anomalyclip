import glob
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"Project root: {PROJECT_ROOT}")


def latest_checkpoint() -> Optional[str]:
    """Find the most recent checkpoint file.
    
    Checks ANOMALY_CLIP_CKPT environment variable first,
    then searches checkpoints/ and logs/ directories.
    Prefers UCF-Crime by default, then ShanghaiTech.
    """
    # Env override
    env_ckpt_path = os.getenv("ANOMALY_CLIP_CKPT")
    if env_ckpt_path:
        ckpt_path = Path(env_ckpt_path)
        if ckpt_path.exists():
            logger.info(f"Using checkpoint from env variable: {ckpt_path}")
            return str(ckpt_path)
        logger.warning(f"ANOMALY_CLIP_CKPT set but file not found: {env_ckpt_path}")

    # Prefer UCF-Crime by default
    ucf_ckpt = PROJECT_ROOT / "checkpoints" / "ucfcrime" / "last.ckpt"
    if ucf_ckpt.exists():
        logger.info(f"Preferring UCF-Crime checkpoint: {ucf_ckpt}")
        return str(ucf_ckpt)

    # Fallback: ShanghaiTech if present
    sht_ckpt = PROJECT_ROOT / "checkpoints" / "shanghaitech" / "last.ckpt"
    if sht_ckpt.exists():
        logger.info(f"Fallback to ShanghaiTech checkpoint: {sht_ckpt}")
        return str(sht_ckpt)

    # Otherwise search newest
    search_patterns = [
        str(PROJECT_ROOT / "checkpoints" / "**" / "*.ckpt"),
        str(PROJECT_ROOT / "logs" / "**" / "*.ckpt"),
    ]

    checkpoint_candidates: List[Tuple[float, str]] = []
    for pattern in search_patterns:
        logger.debug(f"Searching for checkpoints: {pattern}")
        for checkpoint_path in glob.glob(pattern, recursive=True):
            try:
                modification_time = Path(checkpoint_path).stat().st_mtime
                checkpoint_candidates.append((modification_time, checkpoint_path))
            except FileNotFoundError:
                logger.debug(f"Checkpoint file disappeared: {checkpoint_path}")

    if not checkpoint_candidates:
        logger.warning("No checkpoint files found")
        return None

    checkpoint_candidates.sort(key=lambda x: x[0], reverse=True)
    latest_ckpt = checkpoint_candidates[0][1]
    logger.info(f"Found {len(checkpoint_candidates)} checkpoints, using latest: {latest_ckpt}")
    return latest_ckpt


def infer_dataset_from_path(path: str) -> Optional[str]:
    """Infer dataset name from checkpoint path.
    
    Returns one of 'ucfcrime', 'shanghaitech', 'xdviolence', or None.
    """
    path_lower = path.lower()

    if "ucf" in path_lower:
        logger.info("Inferred dataset: ucfcrime")
        return "ucfcrime"
    if "sht" in path_lower or "shanghai" in path_lower:
        logger.info("Inferred dataset: shanghaitech")
        return "shanghaitech"
    if "xd" in path_lower or "xdviolence" in path_lower:
        logger.info("Inferred dataset: xdviolence")
        return "xdviolence"

    logger.warning(f"Could not infer dataset from path: {path}")
    return None


def default_labels_for_dataset(dataset: Optional[str]) -> Optional[str]:
    """Get default labels file for a given dataset.
    
    Order of preference: env override, dataset-specific, UCF->SHT->XD fallback, any labels* CSV.
    """
    # Env override
    env_labels_path = os.getenv("ANOMALY_CLIP_LABELS")
    if env_labels_path:
        labels_path = Path(env_labels_path)
        if labels_path.exists():
            logger.info(f"Using labels from env variable: {labels_path}")
            return str(labels_path)
        logger.warning(f"ANOMALY_CLIP_LABELS set but file not found: {env_labels_path}")

    # Dataset-specific
    labels_file_path: Optional[Path] = None
    if dataset == "ucfcrime":
        labels_file_path = PROJECT_ROOT / "data" / "ucf_labels.csv"
    elif dataset == "shanghaitech":
        labels_file_path = PROJECT_ROOT / "data" / "sht_labels.csv"
    elif dataset == "xdviolence":
        labels_file_path = PROJECT_ROOT / "data" / "xd_labels.csv"

    if labels_file_path and labels_file_path.exists():
        logger.info(f"Found labels file: {labels_file_path}")
        return str(labels_file_path)

    # Fallback order
    for candidate in [PROJECT_ROOT / "data" / "ucf_labels.csv",
                      PROJECT_ROOT / "data" / "sht_labels.csv",
                      PROJECT_ROOT / "data" / "xd_labels.csv"]:
        if candidate.exists():
            logger.info(f"Using fallback labels file: {candidate}")
            return str(candidate)

    # Any labels*.csv
    found_labels = list(PROJECT_ROOT.glob("data/*labels*.csv"))
    if found_labels:
        logger.info(f"Using generic labels file: {found_labels[0]}")
        return str(found_labels[0])

    logger.warning(f"No labels file found for dataset: {dataset}")
    return None


def load_class_names(labels_csv: str) -> List[str]:
    """Load class names from labels CSV file.
    
    Args:
        labels_csv: Path to labels CSV file
        
    Returns:
        List of class names
    """
    logger.info(f"Loading class names from: {labels_csv}")
    try:
        labels_df = pd.read_csv(labels_csv)
        
        if {"id", "name"}.issubset(labels_df.columns):
            # Ensure order by id
            labels_df = labels_df.sort_values("id")
            class_names = labels_df["name"].tolist()
            logger.info(f"Loaded {len(class_names)} class names (sorted by ID)")
            return class_names
        
        # Fallback: use first column
        logger.warning("Labels CSV missing 'id' or 'name' columns, using first column")
        class_names = labels_df.iloc[:, 0].astype(str).tolist()
        logger.info(f"Loaded {len(class_names)} class names from first column")
        return class_names
        
    except Exception as e:
        logger.error(f"Failed to load class names from {labels_csv}: {e}")
        raise


def normal_index_from_names(names: List[str]) -> int:
    """Find the index of the 'normal' class in class names.
    
    Args:
        names: List of class names
        
    Returns:
        Index of 'normal' class, or 0 if not found
    """
    for index, class_name in enumerate(names):
        if class_name.strip().lower() == "normal":
            logger.info(f"Found 'normal' class at index: {index}")
            return index
    
    logger.warning("'normal' class not found in labels, using index 0 as default")
    return 0


def discover_runtime() -> Tuple[str, str, str, int]:
    """Auto-discover runtime configuration for AnomalyCLIP.
    
    Finds checkpoint, labels file, architecture, and normal class index.
    
    Returns:
        Tuple of (checkpoint_path, labels_file, architecture, normal_class_id)
        
    Raises:
        RuntimeError: If checkpoint or labels cannot be found
    """
    logger.info("Discovering runtime configuration...")
    
    # Find checkpoint
    checkpoint_path = latest_checkpoint()
    if not checkpoint_path:
        error_msg = (
            "No checkpoint found. Either set ANOMALY_CLIP_CKPT environment variable "
            "or place a .ckpt file in checkpoints/ or logs/ directory."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Infer dataset and find labels
    inferred_dataset = infer_dataset_from_path(checkpoint_path)
    labels_file = default_labels_for_dataset(inferred_dataset)
    
    if not labels_file:
        error_msg = (
            "No labels CSV found. Either set ANOMALY_CLIP_LABELS environment variable "
            "or add a labels CSV file to data/ directory."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Load class names and find normal class
    class_names = load_class_names(labels_file)
    normal_class_id = normal_index_from_names(class_names)
    
    # Get architecture
    architecture = os.getenv("ANOMALY_CLIP_ARCH", "ViT-B/16")
    logger.info(f"Using architecture: {architecture}")
    
    logger.info(
        f"Runtime config discovered - checkpoint: {Path(checkpoint_path).name}, "
        f"labels: {Path(labels_file).name}, arch: {architecture}, normal_id: {normal_class_id}"
    )
    
    return checkpoint_path, labels_file, architecture, normal_class_id
