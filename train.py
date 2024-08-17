from visionik.dataset import IKDataset
from visionik.model import VisionIKModel, ShuffleNet_V2_X0_5, MobileNet_V3_Large, MobileNet_V3_Small 
from visionik.trainer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import models

MODELS = {
    "VisionIKModel": VisionIKModel,
    "ShuffleNet_V2_X0_5": ShuffleNet_V2_X0_5,
    "MobileNet_V3_Large": MobileNet_V3_Large,
    "MobileNet_V3_Small": MobileNet_V3_Small
}

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configuration to verify it's loaded correctly
    print(OmegaConf.to_yaml(cfg))

    # Initialize the model
    model = MODELS[cfg.model.name](cfg.model)

    # Initialize the trainer with the model and configuration
    trainer = Trainer(model=model, cfg=cfg)

    # Start the training process
    trainer.train()

    # Save the trained model
    trainer.save_model(cfg.trainer.model_checkpoint_path)

if __name__ == "__main__":
    main()
