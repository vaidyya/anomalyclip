
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule

from src import utils
from src.utils.augmentations import get_augmentations as get_transforms
from src.models.anomaly_clip_module import AnomalyCLIPModule

log = utils.get_pylogger(__name__)


class LiveStreamDataset:
    def __init__(self, transform=None, video_path=None):
        self.transform = transform
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        if self.transform:
            image = self.transform(image)
        return image

    def release(self):
        self.cap.release()


def live_analysis(model: AnomalyCLIPModule, video_path: str = None):
    """Performs live anomaly detection on a webcam feed."""

    model.eval()

    transform = get_transforms(224, 1)

    dataset = LiveStreamDataset(transform=transform, video_path=video_path)

    class_names = model.trainer.datamodule.class_names
    normal_id = model.trainer.datamodule.hparams.normal_id

    while True:
        try:
            frame = next(dataset)
            frame = frame.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                similarity, abnormal_scores = model.forward(
                    frame,
                    labels=torch.zeros(1),
                    ncentroid=model.ncentroid,
                    segment_size=1,
                    test_mode=True,
                )
                prob = torch.sigmoid(abnormal_scores).item()
                
                # Compute class probabilities
                softmax_similarity = torch.softmax(similarity, dim=1)
                class_probs = softmax_similarity * abnormal_scores.unsqueeze(1)
                
                # Get top classes
                top_probs, top_indices = torch.topk(class_probs.squeeze(), k=3)
                top_classes = [{"name": class_names[i], "prob": p.item()} for i, p in zip(top_indices, top_probs)]


                yield {
                    "score": abnormal_scores.item(),
                    "prob": prob,
                    "class_probabilities": {
                        "labels": [c for i, c in enumerate(class_names) if i != normal_id],
                        "data": class_probs.squeeze().tolist(),
                    },
                    "top_classes": top_classes,
                }

        except StopIteration:
            break

    dataset.release()


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    live_analysis(cfg)


if __name__ == "__main__":
    main()
