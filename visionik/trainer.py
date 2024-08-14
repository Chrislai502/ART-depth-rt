import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import neptune
from neptune.utils import stringify_unsupported
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from visionik.model import VisionIKModel
from visionik.dataset import IKDataset

class Trainer:
    def __init__(self, model, cfg: DictConfig, resume_checkpoint=None):
        self.model = model
        self.epochs = cfg.trainer.epochs
        self.checkpoint_interval = cfg.trainer.checkpoint_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load Dataset
        dataset_cfg = cfg.dataset
        train_dataset = IKDataset(
            dataset_dir=os.path.join(dataset_cfg.dataset_dir, 'train'),
            save_plots=dataset_cfg.save_plots,
            stat_percentile_range=dataset_cfg.stat_percentile_range,
            cfg=cfg.dataset
        )
        val_dataset = IKDataset(
            dataset_dir=os.path.join(dataset_cfg.dataset_dir, 'val'),
            cfg=cfg.dataset,
            val = True
        )

        self.dataset_mean_action_value = train_dataset.mean_action_value
        self.dataset_std_action_value = train_dataset.std_action_value

        # Train-validation split
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=cfg.trainer.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.learning_rate)

        self.start_epoch = 0
        self.step = 0

        # Initialize Neptune logger
        self.run = neptune.init_run(
            project=cfg.trainer.project_name, 
            api_token=cfg.neptune.api_token
        )

        # Log hyperparameters
        self.run["hyperparameters"] = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))

        # Resume from checkpoint if specified
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    def train(self):
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0

            # Wrap the training DataLoader with tqdm
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{self.epochs}")
                
                for i, (images, labels) in enumerate(tepoch):
                    self.step += 1

                    # Normalize the labels
                    labels = (labels - self.dataset_mean_action_value) / self.dataset_std_action_value

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
                    self.run["train/batch_loss"].append(loss.item(), step=self.step)

                    # Checkpoint the model every `n` steps
                    if self.step % self.checkpoint_interval == 0:
                        checkpoint_path = f"checkpoint_epoch_{epoch}_step_{self.step}.pth"
                        self.save_checkpoint(checkpoint_path)

            # Log epoch loss to Neptune
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.run["train/epoch_loss"].append(avg_epoch_loss, step=self.step)

            # Validate the model after each epoch
            val_loss = self.validate()
            self.run["validation/epoch_loss"].append(val_loss, step=self.step)

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        print("Training complete.")

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = (labels - self.dataset_mean_action_value) / self.dataset_std_action_value
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'step': self.step,
            'mean': self.dataset_mean_action_value,
            'std': self.dataset_std_action_value
        }
        torch.save(checkpoint, path)
        self.run[f"ckpt/{path.split('/')[-1]}"].upload(path)
        print(f"Checkpoint saved to {path} and uploaded to Neptune.")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.full_dataset.mean_action_value = checkpoint['mean']
        self.full_dataset.std_action_value = checkpoint['std']
        print(f"Resumed training from checkpoint {path}.")

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

    # Check if resuming from a checkpoint
    resume_checkpoint = cfg.trainer.resume_checkpoint if "resume_checkpoint" in cfg.trainer else None

    # Initialize trainer
    trainer = Trainer(model=model, cfg=cfg, resume_checkpoint=resume_checkpoint)

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(cfg.trainer.model_checkpoint_path)

if __name__ == "__main__":
    main()
