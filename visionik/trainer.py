import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import neptune
from neptune.utils import stringify_unsupported
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from visionik.model import VisionIKModel
from visionik.dataset import IKDataset

class Trainer:
    def __init__(self, model, cfg: DictConfig):
        self.model = model
        self.epochs = cfg.trainer.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load Dataset
        dataset_cfg = cfg.dataset
        full_dataset = IKDataset(
            dataset_dir=dataset_cfg.dataset_dir,
            save_plots=dataset_cfg.save_plots,
            stat_percentile_range=dataset_cfg.stat_percentile_range
        )

        # Train-validation split
        train_val_split = cfg.trainer.train_val_split
        train_size = int(train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.learning_rate)
        
        # Initialize Neptune logger
        self.run = neptune.init_run(
            project=cfg.trainer.project_name, 
            api_token=cfg.neptune.api_token
        )

        # Log hyperparameters
        self.run["hyperparameters"] = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Log batch loss to Neptune
                self.run["train/batch_loss"].log(loss.item())

            # Log epoch loss to Neptune
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.run["train/epoch_loss"].log(avg_epoch_loss)

            # Validate the model after each epoch
            val_loss = self.validate()
            self.run["validation/epoch_loss"].log(val_loss)

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        print("Training complete.")

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def save_model(self, path=None):
        if path is None:
            path = "vision_ik_model.pth"
        torch.save(self.model.state_dict(), path)
        self.run["model_checkpoint"].upload(path)
        print(f"Model saved to {path} and uploaded to Neptune.")

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize model
    model = VisionIKModel()

    # Initialize trainer
    trainer = Trainer(model=model, cfg=cfg)

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(cfg.trainer.model_checkpoint_path)

if __name__ == "__main__":
    main()
