from visionik.dataset import IKDataset
from visionik.model import VisionIKModel
from visionik.trainer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configuration to verify it's loaded correctly
    print(OmegaConf.to_yaml(cfg))

    # Initialize the model
    model = VisionIKModel()

    # Initialize the trainer with the model and configuration
    trainer = Trainer(model=model, cfg=cfg)

    # Start the training process
    trainer.train()

    # Save the trained model
    trainer.save_model(cfg.trainer.model_checkpoint_path)

if __name__ == "__main__":
    main()
