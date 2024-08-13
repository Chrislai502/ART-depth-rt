import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from visionik.dataset import IKDataset
import hydra
from omegaconf import DictConfig, OmegaConf

def test_ik_dataset(cfg: DictConfig):
    
    # Create dataset
    ik_dataset = IKDataset(
        dataset_dir=cfg.dataset.dataset_dir,
        save_plots=cfg.dataset.save_plots,
        stat_percentile_range=cfg.dataset.stat_percentile_range,
        cfg=cfg.dataset
    )

    # Create data loader
    dataloader = DataLoader(ik_dataset, batch_size=5, shuffle=True, num_workers=2)

    # Iterate over the DataLoader
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Image batch shape: {images.size()}")
        print(f"Label batch shape: {labels.size()}")
        print(f"Labels: {labels}")

        # Display and save the images side by side
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for j in range(images.size(0)):
            image = images[j].permute(1, 2, 0).numpy()
            axes[j].imshow(image)
            axes[j].set_title(f"Label: {labels[j].numpy()}")
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'batch_{i+1}.png')
        plt.show()
        
        break

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    test_ik_dataset(cfg=cfg)

if __name__ == "__main__":
    main()
