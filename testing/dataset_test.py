import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from visionik.dataset import IKDataset

def test_ik_dataset():
    
    # Create dataset
    dataset = IKDataset('../datasets/kitting_vision_ik/')

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Iterate over the DataLoader
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Image batch shape: {images.size()}")
        print(f"Label batch shape: {labels.size()}")
        print(f"Labels: {labels}")

        # Display the images in the batch
        for j in range(images.size(0)):
            image = images[j].permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.title(f"Sample {j+1} - Label: {labels[j]}")
            plt.show()

        break

test_ik_dataset()
