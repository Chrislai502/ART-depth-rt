import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import v2

# Define the path to your image
image_path = "second_seg_first_frame.png"

# Load the image
image = Image.open(image_path).convert('RGB')

# Get Image size
image_size = image.size
print(f"Image size: {image_size}")
resize_factor = 0.95

# Define the augmentations
augmentations = {
    # "Original": transforms.Compose([]),
    # "ColorJitter": transforms.ColorJitter(brightness=0.15, contrast=0.5, saturation=0.5, hue=0.1),
    # "RandomAdjustSharpness": transforms.RandomAdjustSharpness(sharpness_factor=2.0),
    # "GaussianNoise": v2.GaussianNoise(mean=0.0, sigma=0.1), # For Pil Images
    # "ColorJitter": transforms.ColorJitter(brightness=0.15, contrast=0.5, saturation=0.5, hue=0.1),
    # "RandomPerspective": transforms.RandomPerspective(distortion_scale=0.2, p=0.8),
    # "ElasticTransform": transforms.ElasticTransform(alpha=300.0, sigma=7.0), # Sigma correlates to image clarity, alpha correlates to waviness.
    # "RandomResizedCrop": transforms.RandomResizedCrop(size=(int((resize_factor*image_size[1])), int(resize_factor*image_size[0])), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    "JPEGCompression": v2.JPEG((5, 60)),
    # "Resize": transforms.Resize(size=(48, 64))
}

# Apply each augmentation and display/save the result
plt.figure(figsize=(15, 3))  # Adjust the figure size to accommodate multiple images

for i in range(5):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    for name, augmentation in augmentations.items():
        image = augmentation(image)
            
    # Display the image
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(f"{name} {i + 1}")
    plt.axis('off')

plt.suptitle(f'{name} Augmentation Examples')

# Save the entire figure as a single image
combined_save_path = f"augmented_{name}_combined.jpg"
combined_save_path = f"jpeg resize_combined.jpg"
plt.savefig(combined_save_path)
print(f"Saved combined {name} image to {combined_save_path}")

plt.show()