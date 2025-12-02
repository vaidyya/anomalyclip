
import base64
import os
import tempfile
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import hydra
from omegaconf import DictConfig
import pyrootutils

from src.live_analysis import live_analysis

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

from pytorch_lightning import LightningModule
from src.live_analysis import live_analysis

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = None


@app.get("/")
async def read_root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        video_data = base64.b64decode(data["video"].split(",")[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_data)
            video_path = temp_video.name

        for result in live_analysis(model, video_path):
            await websocket.send_json(result)

        os.remove(video_path)


def run_app(hydra_cfg: DictConfig) -> None:
    global model
    
    # Instantiate model and datamodule
    model: LightningModule = hydra.utils.instantiate(hydra_cfg.model)
    datamodule = hydra.utils.instantiate(hydra_cfg.data)
    datamodule.setup()
    model.trainer = hydra.utils.instantiate(hydra_cfg.trainer, logger=False)
    model.trainer.datamodule = datamodule

    # Construct absolute path to checkpoint
    if hydra_cfg.ckpt_path and not os.path.isabs(hydra_cfg.ckpt_path):
        hydra_cfg.ckpt_path = os.path.join(os.environ["PROJECT_ROOT"], hydra_cfg.ckpt_path)
    
    # Load ncentroid
    ckpt_path = Path(hydra_cfg.ckpt_path)
    save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
    save_dir = Path(os.path.join(os.environ["PROJECT_ROOT"], "logs", "train", "runs", str(save_dir)))
    ncentroid_file = Path(save_dir / "ncentroid.pt")
    if ncentroid_file.is_file():
        model.ncentroid = torch.load(ncentroid_file)
    else:
        raise FileNotFoundError(f"ncentroid file {ncentroid_file} not found")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@hydra.main(version_base="1.3", config_path="./configs", config_name="eval.yaml")
def main(hydra_cfg: DictConfig) -> None:
    run_app(hydra_cfg)


if __name__ == "__main__":
    # Set the project root
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    os.environ["PROJECT_ROOT"] = str(path)
    main()
