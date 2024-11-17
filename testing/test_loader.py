import argparse
from artdepth.utils.config import get_config
from artdepth.dataset import MixedARTKITTINYU, DepthDataLoader
from pprint import pprint
import torch
import os

import matplotlib.pyplot as plt  # Added for displaying images
import torchvision.transforms as T  # For unnormalizing tensors
import hydra
from omegaconf import DictConfig, OmegaConf


# Set environment variables for OpenGL and WandB (if needed)
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): random seed
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def vstack_batch_images(batch_images):
    """
    Vertically stack a batch of images.

    Args:
        batch_images (torch.Tensor): A batch of images with shape (B, C, H, W).
    
    Returns:
        torch.Tensor: A single image tensor with all images vertically stacked.
    """
    # Assuming the batch is a tensor of shape (B, C, H, W)
    return torch.cat([img for img in batch_images], dim=1)  # Stack along height (dim=1)

def show_vstacked_images(batch_images):
    """
    Display a batch of images vertically stacked using matplotlib.

    Args:
        batch_images (torch.Tensor): A batch of images with shape (B, C, H, W).
    """
    # Unnormalize each image in the batch
    unnormalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    # batch_images = torch.stack([unnormalize(img) for img in batch_images])  # Unnormalize all images
    batch_images = torch.stack([img for img in batch_images])  # Unnormalize all images
    
    batch_images = torch.clamp(batch_images, 0, 1)  # Clip values to [0, 1]

    # Vertically stack all images in the batch
    vstacked_image = vstack_batch_images(batch_images)

    # Convert to NumPy for displaying
    vstacked_image_np = vstacked_image.permute(1, 2, 0).cpu().numpy()

    # Display using matplotlib
    plt.imshow(vstacked_image_np)
    plt.axis('off')
    plt.show(block=False)  # Show without blocking execution
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close()  # Close the window after key press

def test_data_loading(config, dataset_class, sample_ratio=None, mode="train"):
    """
    Test loading and iterating over a dataset.

    Args:
        config: Configuration object for the dataset.
        dataset_class: Dataset class to be tested (MixedNYUKITTI or MixedARTKITTINYU).
        mode (str): Mode for data loading, e.g., "train" or "online_eval".
    """
    print(f"\nTesting {dataset_class.__name__} in '{mode}' mode")

    # Initialize the dataset loader
    if sample_ratio:
        # data_loader = DepthDataLoader(config.dataset, mode, device=device).data
        
        data_loader = dataset_class(config.dataset, mode, sample_ratio).data
    else:
        data_loader = DepthDataLoader(config.dataset, mode, device=device).data

    # Iterate through batches to test data loading
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}")
        
        # Display the vertically stacked images of the batch
        if 'image' in batch:
            print(f"  Displaying vertically stacked images from batch {i + 1}")
            images = batch['image']  # Entire batch of images (B, C, H, W)
            show_vstacked_images(images)

        # Log shapes of other batch data
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape {value.shape}")

        # Press a key to proceed to the next batch
        print("Press a key to proceed to the next batch.")
        # break  # Remove this break to process all batches

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test data loading for MixedARTKITTINYU datasets.")
    parser.add_argument("-m", "--model", type=str, default="zoedepth", help="Model name (default: synunet)")
    parser.add_argument("-d", "--dataset", type=str, default='mix', help="Dataset name (default: mix)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing (default: 4)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading (default: 2)")
    parser.add_argument("-o", "--mode", type=str, default="train", help="train or online_eval")

    args, unknown_args = parser.parse_known_args()
    
    fix_random_seed(43)
    
    print(OmegaConf.to_yaml(cfg))
    test_data_loading(cfg, MixedARTKITTINYU, {'art': 1, 'kitti': 1}, mode = args.mode)
    # test_data_loading(config=cfg, dataset_class=DepthDataLoader, mode=args.mode)

if __name__ == "__main__":
    main()
