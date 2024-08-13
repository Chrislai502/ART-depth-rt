import torch
from visionik.model import VisionIKModel
from visionik.dataset import IKDataset
from torchvision import transforms
from PIL import Image
import os

class VisionIKInference:
    def __init__(self, checkpoint_path):
        # Load the checkpoint when initializing the class
        self.model, self.mean, self.std = self.load_checkpoint(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        model = VisionIKModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, checkpoint['mean'], checkpoint['std']

    def unnormalize(self, output):
        return output * self.std + self.mean

    def preprocess_image(self, image):
        # Load and preprocess the image
        # image = IKDataset.transform_image(image_path)  # Assuming you have a similar function in IKDataset
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def run_inference(self, image):
        image = self.preprocess_image(image)
        # Run inference
        with torch.inference_mode():
            output = self.model(image)
        
        # Unnormalize the output
        output = output.to('cpu').squeeze(0).detach().numpy()
        output = self.unnormalize(output)
        
        return output
    
    def infer_gif(self, gif_path):
        
        infer_outputs = []
        # Load and Parse the gif. For every frame, run inference
        with Image.open(gif_path) as img:
            frame_count = 0
            while True:
                # Convert Image to Tensor
                # Convert Image to RGB
                image = img.convert('RGB')
                image = self.model.preprocess_img(image)
                image = image.to(self.device)
                output = self.run_inference(image)

                infer_outputs.append(output)

                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break
                frame_count += 1
        return infer_outputs

        


if __name__ == "__main__":
    image_path = "/path/to/image.png"
    checkpoint_path = "/path/to/checkpoint.pth"
    
    inference_engine = VisionIKInference(checkpoint_path)
    output = inference_engine.run_inference(image_path)
    print(f"Inference Output: {output}")
