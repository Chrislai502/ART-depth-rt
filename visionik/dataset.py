import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import os

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


class IKDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, stat_percentile_range=(1, 99)):
        '''
        Args:
            image_dir (string): Directory with all the images.
            labels_file (string): Path to the file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        image_dir = os.path.join(dataset_dir, 'images')
        self.image_dir = image_dir
        self.labels = self.load_labels(os.path.join(dataset_dir, 'labels.csv'))
        self.transform = transform or transforms.Compose([
            # transforms.Resize((62, 42)),
            transforms.ToTensor()
        ])

        # Calculate Statistics
        print('Calculating Statistics...')
        self.stat_percentile_range = stat_percentile_range
        self.max_action_value, self.min_action_value = self.calculate_min_max()
        self.mean_action_value, self.std_action_value = self.calculate_mean_std()

    def calculate_mean_std(self):
        all_labels = np.array(list(self.labels.values()))

        # Remove Percentile Outliers
        lower_percentile = np.percentile(all_labels, self.stat_percentile_range[0], axis=0)
        upper_percentile = np.percentile(all_labels, self.stat_percentile_range[1], axis=0)

        # Remove Outliers
        trimmed_labels = all_labels[(all_labels >= lower_percentile) & (all_labels <= upper_percentile)]

        mean_action_value = np.mean(trimmed_labels, axis=0)
        std_action_value = np.std(trimmed_labels, axis=0)

        return mean_action_value, std_action_value

    def calculate_min_max(self):
        all_labels = np.array(list(self.labels.values()))
        max_action_value = np.max(all_labels, axis=0)
        min_action_value = np.min(all_labels, axis=0)
        return max_action_value, min_action_value

    def load_labels(self, labels_file):
        '''
        Load the labels from a file. Assumes a CSV file with image filenames and corresponding lables.
        '''
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                filename = parts[0]
                label = np.array([float(x) for x in parts[1:-1]], dtype=np.float32)
                labels[filename] = label
        return labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')

        if self.transform: # Apply the transform if it exists
            image = self.transform(image)

        label = self.labels[img_name]
        
        # Normalize the label
        label = (label - self.mean_action_value) / self.std_action_value
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label