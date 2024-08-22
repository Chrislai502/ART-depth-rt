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

**Note by Chris:** Training with resumption feature is still not functional via `resume_checkpoint` flag in the config. This is because Neptune expects local run metadata to be stored in `.neptune/` to be at root dir of program. However, `hydra` seems to have inherently directed all created files into `outputs/`, which is clean, but breaks this functionality. To be figured out by the next person!

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

Also refer to the `vidgen_kitting_demo.py` from the [mobile-aloha](https://github.com/CollaborativeRobotics/mobile-aloha/blob/chris/vidgen_demo_kitting/aloha_scripts/vidgen_kitting_demo.py) repository for use of inferrence with gifs.

## Model Checkpoint

The model checkpoint contains:
- `model_state_dict`: Model State dictionary
- `model_cfg`: hydra OmegaConf container to create model using hydra.
- `optimizer_state_dict`: Optimizer state dictionary
- `epoch`: Epochs trained
- `step`: Steps trained
- `mean`: The mean joint positions of the dataset this model is trained on
- `std`: The std joint positions of the dataset this model is trained on
- `neptune_run_id`: Neptune Run ID for resuming.


The `VisionIKInference` class automatically loads these parameters when initialized.


## Notes

- The script supports both CPU and GPU inference.


---

Feel free to reach out with any questions to Chris @ cl@co.bot

