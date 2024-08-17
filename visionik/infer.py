import torch
from visionik.model import VisionIKModel, MobileNet_V3_Small
from visionik.dataset import IKDataset
from torchvision import transforms
from PIL import Image
from scipy.interpolate import splrep, splev # B Spline
from scipy.interpolate import UnivariateSpline # Univariate Spline
import numpy as np
import os

class VisionIKInference:
    def __init__(self, checkpoint_path):
        # Load the checkpoint when initializing the class
        self.model, self.mean, self.std = self.load_checkpoint(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        model = MobileNet_V3_Small()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, checkpoint['mean'], checkpoint['std']

    def unnormalize(self, output):
        return output * self.std + self.mean

    def preprocess_image(self, image):
        # Load and preprocess the image
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        return image

    def interpolate_trajectory(self, infer_outputs, type="Univariate", num_points=100):
        infer_outputs = np.array(infer_outputs)
        num_frames, num_dim = infer_outputs.shape
        output_waypoints = np.zeros((num_points, num_dim))

        for dim_idx in range(num_dim):
            if type == "Univariate":
                t_vals = np.linspace(0, 1, num_points)
                us = UnivariateSpline(t_vals, infer_outputs[:, dim_idx], k=5)
                output_waypoints[:, dim_idx] = us(t_vals)         
            # elif type == "BSpline":
            # elif type == "Bezier":
            else:
                raise ValueError("Invalid Interpolation Type")
        return output_waypoints

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
