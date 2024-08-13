import torch
from visionik.model import VisionIKModel
from visionik.dataset import IKDataset
import os

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = VisionIKModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['mean'], checkpoint['std']

def unnormalize(output, mean, std):
    return output * std + mean

def inference(image_path, checkpoint_path):
    # Load the checkpoint
    model, mean, std = load_checkpoint(checkpoint_path)

    # Load and preprocess the image
    image = IKDataset.transform_image(image_path)  # Assuming you have a similar function in IKDataset
    image = image.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(image)
    
    # Unnormalize the output
    output = unnormalize(output, mean, std)
    
    return output

if __name__ == "__main__":
    image_path = "/path/to/image.png"
    checkpoint_path = "/path/to/checkpoint.pth"
    
    output = inference(image_path, checkpoint_path)
    print(f"Inference Output: {output}")
