import torch
from art-depth.model import MobileNet_V3_Small
from torchvision.transforms import transforms
from PIL import Image
import hydra

class ARTDepthInference:
    def __init__(self, checkpoint_path, cfg = None):
        # Load the checkpoint when initializing the class
        self.cfg = cfg  # Store the configuration
        self.model, self.mean, self.std = self.load_checkpoint(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        if self.cfg:
            model = hydra.utils.instantiate(self.cfg.model)
        else:
            model = MobileNet_V3_Small()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, checkpoint['mean'], checkpoint['std']

    def unnormalize(self, output):
        return output * self.std + self.mean

    def preprocess_image(self, image):
        # Load and preprocess the image
        # image = IKDataset.transform_image(image_path)  # Assuming you have a similar function in IKDataset
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
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
    image_path = "../duck.jpg"
    checkpoint_path = "/home/cobot/testing/Vision-ik/outputs/2024-08-22/21-43-21/checkpoint_epoch_0_step_1000.pth"
    image = Image.open(image_path)

    inference_engine = ARTDepthInference(checkpoint_path)
    output = inference_engine.run_inference(image)
    print(f"Inference Output: {output}")
