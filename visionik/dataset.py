import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
ALL_NAMES = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]

class IKDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, stat_percentile_range=(1, 99), save_plots=True):
        '''
        Args:
            dataset_dir (string): Directory with all the dataset files.
            transform (callable, optional): Optional transform to be applied on a sample.
            stat_percentile_range (tuple): Percentile range for trimming the dataset for mean/std calculation.
            save_plots (bool): Whether to save distribution plots.
            plot_dir (str): Directory where the plots will be saved.
        '''
        image_dir = os.path.join(dataset_dir, 'images')
        self.image_dir = image_dir
        self.labels = self.load_labels(os.path.join(dataset_dir, 'labels.csv'))
        self.transform = transform or transforms.Compose([
            transforms.Resize((48, 64)),
            transforms.ToTensor()
        ])
        self.stat_percentile_range = stat_percentile_range
        self.save_plots = save_plots
        self.plot_dir = os.path.join(dataset_dir, 'stat_plots')

        if self.save_plots and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Calculate Statistics
        print('Calculating Statistics...')
        self.max_action_value, self.min_action_value = self.calculate_min_max()
        self.mean_action_value, self.std_action_value = self.calculate_mean_std()

    def calculate_min_max(self):
        all_labels = np.array(list(self.labels.values()))
        max_action_value = np.max(all_labels, axis=0)
        min_action_value = np.min(all_labels, axis=0)
        return max_action_value, min_action_value

    def calculate_mean_std(self):
        all_labels = np.array(list(self.labels.values()))

        # Remove Percentile Outliers
        lower_percentile = np.percentile(all_labels, self.stat_percentile_range[0], axis=0)
        upper_percentile = np.percentile(all_labels, self.stat_percentile_range[1], axis=0)

        # Remove Outliers (ensure proper indexing)
        mask = np.all((all_labels >= lower_percentile) & (all_labels <= upper_percentile), axis=1)
        trimmed_labels = all_labels[mask]

        mean_action_value = np.mean(trimmed_labels, axis=0)
        std_action_value = np.std(trimmed_labels, axis=0)

        if self.save_plots:
            self.plot_distributions(all_labels, trimmed_labels, mean_action_value, std_action_value)

        return mean_action_value, std_action_value

    def plot_distributions(self, all_labels, trimmed_labels, mean_values, std_values):
        for i in range(len(ALL_NAMES)):
            plt.figure(figsize=(12, 6))

            # Plot original distribution
            plt.subplot(1, 2, 1)
            plt.hist(all_labels[:, i], bins=50, alpha=0.7, color='blue')
            plt.axvline(x=np.mean(all_labels[:, i]), color='red', linestyle='--', label=f'Mean: {np.mean(all_labels[:, i]):.2f}')
            plt.axvline(x=np.mean(all_labels[:, i]) + np.std(all_labels[:, i]), color='green', linestyle='--', label=f'Std Dev: {np.std(all_labels[:, i]):.2f}')
            plt.axvline(x=np.mean(all_labels[:, i]) - np.std(all_labels[:, i]), color='green', linestyle='--')
            plt.title(f'{ALL_NAMES[i]} - Original')
            plt.legend()

            # Plot trimmed distribution
            plt.subplot(1, 2, 2)
            plt.hist(trimmed_labels[:, i], bins=50, alpha=0.7, color='orange')
            plt.axvline(x=mean_values[i], color='red', linestyle='--', label=f'Mean: {mean_values[i]:.2f}')
            plt.axvline(x=mean_values[i] + std_values[i], color='green', linestyle='--', label=f'Std Dev: {std_values[i]:.2f}')
            plt.axvline(x=mean_values[i] - std_values[i], color='green', linestyle='--')
            plt.title(f'{ALL_NAMES[i]} - Trimmed')
            plt.legend()

            # Save plot
            plt.suptitle(f'Distribution of {ALL_NAMES[i]}')
            plt.savefig(os.path.join(self.plot_dir, f'{ALL_NAMES[i]}_distribution.png'))
            plt.close()

    def load_labels(self, labels_file):
        '''
        Load the labels from a file. Assumes a CSV file with image filenames and corresponding labels.
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

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_name]
        
        # Normalize the label
        label = (label - self.mean_action_value) / self.std_action_value
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
