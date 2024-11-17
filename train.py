from art-depth.trainer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configuration to verify it's loaded correctly
    print(OmegaConf.to_yaml(cfg))

    # Initialize the trainer with the model and configuration
    trainer = Trainer(cfg=cfg)

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()
