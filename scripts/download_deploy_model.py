import wandb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

root = Path(__file__).parent.parent
deploy_model_path = root / "mush_app" / "model"
deploy_model_path.mkdir(parents=True, exist_ok=False)

wandb.init(
    project="Mushroom-Classifier", name="download_model", job_type="download_model", 
    dir=deploy_model_path, save_code=False, resume="allow", tags=["download_model"]
)

artifact = wandb.use_artifact('g-broughton/model-registry/Mushroom-Classifier:latest', type='model').download()