# Vision-ik
Vision based IK model on the Aloha arms platform.

## Project Structure

- `visionik/`: Contains the `VisionIKModel` and `IKDataset` classes.
- `infer.py`: The inference script described in this README.

## Setup

### 1. Creating the Environment
For inference Setup on the mobile-aloha platform, create an environment using `environment_robot.yaml`. Note, `JPEGCompression` data augmentation is not supported in the earlier torchvision included in `environment_robot.yaml`, so comment the cinfiguration lines out in `conf/config.yaml`.

```
conda env create -f environment_robot.yaml
```

### 2. Install the Package
```
pip install -e .
```

## Training
1. Copy and rename `sample_config.yaml` to `config.yaml` and fill out:
    1. `dataset_dir`: "\<Absolute path to dataset directory\>"
    2. `api_token`: "\<Neptune API Token\>"
2. Run the Training Script:
    ```
    python3 train.py
    ```

## Inference

### 1. Loading the Model and Running Inference

The `VisionIKInference` class is designed to load a pre-trained model from a checkpoint and run inference on images or GIFs.

**Example Usage:**

```python
from visionik.inference import VisionIKInference
from PIL import Image

checkpoint_path = "/path/to/checkpoint.pth"
image_path = "/path/to/image.png"
image = Image.open(image_path)


# Initialize the inference engine
inference_engine = VisionIKInference(checkpoint_path)

# Run inference on a single image
output = inference_engine.run_inference(image)
print(f"Inference Output: {output}")
```

### 2. Inference on Single Images

The `run_inference(image)` method handles preprocessing, model inference, and unnormalizing the output to provide the final prediction.

**Input:** 
- A preprocessed image tensor.

**Output:** 
- The unnormalized output vector representing the IK solution.

### 3. Inference on GIFs

The `infer_gif(gif_path)` method processes every frame in a GIF, runs inference on each frame, and returns a list of predictions.

**Example Usage:**

```python
gif_path = "/path/to/your/animation.gif"
outputs = inference_engine.infer_gif(gif_path)
for i, output in enumerate(outputs):
    print(f"Frame {i} Output: {output}")
```

Also refer to 

### 4. Model Checkpoint

The model checkpoint is expected to contain:
- `model_state_dict`: The state dictionary for the VisionIK model.
- `mean`: Mean of the dataset used for normalization.
- `std`: Standard deviation of the dataset used for normalization.

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'step': self.step,
            'mean': self.dataset_mean_action_value,
            'std': self.dataset_std_action_value
        }

The `VisionIKInference` class automatically loads these parameters when initialized.

### 5. Preprocessing and Unnormalizing

The script applies preprocessing to input images and unnormalizes the output using the stored dataset statistics (mean and standard deviation). Make sure your dataset preprocessing aligns with the following expectations:

- Input images should be in RGB format.
- Preprocessing includes resizing and normalizing the images.

### 6. Running the Script

To run the inference directly from the command line, update the paths in the script:

```bash
python inference.py
```

Make sure to replace:

- `/path/to/checkpoint.pth` with the path to your trained model checkpoint.
- `/path/to/image.png` with the image you want to run inference on.

## Notes

- The inference script assumes the presence of specific methods in the `IKDataset` and `VisionIKModel` classes, such as `transform_image` and `preprocess_img`. Customize these methods as needed based on your implementation.
- The script supports both CPU and GPU inference.


---

Feel free to reach out with any questions to Chris @ cl@co.bot

