import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
ALL_NAMES = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]

class IKDataset(Dataset):
    def __init__(self, dataset_dir, cfg : DictConfig, stat_percentile_range=(1, 99), save_plots=False, val = False):
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
        first_image_path = os.path.join(self.image_dir, os.listdir(self.image_dir)[0])
        with Image.open(first_image_path) as img:
            self.image_size = img.size  # (width, height)
        self.labels = self.load_labels(os.path.join(dataset_dir, 'labels.csv'))
        self.stat_percentile_range = stat_percentile_range
        apply_augmentations = cfg.augmentations.get("apply", False)
        if val:
            apply_augmentations = False
            stat_percentile_range = (0, 100)

        # Augmentations
        augmentation_transforms = []
        augmentations = cfg.augmentations
        if augmentations and apply_augmentations:
            aug_cfg = augmentations

            if "ColorJitter" in aug_cfg:
                cj = aug_cfg["ColorJitter"]
                augmentation_transforms.append(transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=cj.get("brightness", 0.15),
                        contrast=cj.get("contrast", 0.5),
                        saturation=cj.get("saturation", 0.5),
                        hue=cj.get("hue", 0.1)
                    )],
                    p=cj.get("p", 0.5)
                ))

            if "RandomPerspective" in aug_cfg:
                rp = aug_cfg["RandomPerspective"]
                augmentation_transforms.append(transforms.RandomApply(
                    [transforms.RandomPerspective(
                        distortion_scale=rp.get("distortion_scale", 0.2),
                        p=rp.get("p", 0.8)
                    )],
                    p=rp.get("p", 0.8)
                ))

            if "ElasticTransform" in aug_cfg:
                et = aug_cfg["ElasticTransform"]
                augmentation_transforms.append(transforms.RandomApply(
                    [transforms.ElasticTransform(
                        alpha=et.get("alpha", 300.0),
                        sigma=et.get("sigma", 7.0)
                    )],
                    p=et.get("p", 0.5)
                ))

            if "RandomResizedCrop" in aug_cfg:
                rrc = aug_cfg["RandomResizedCrop"]
                resize_factor = rrc["crop_scale"]
                rrc_size = (int((resize_factor*self.image_size[1])), int(resize_factor*self.image_size[0]))
                augmentation_transforms.append(transforms.RandomApply(
                    [transforms.RandomResizedCrop(
                        size=tuple(rrc_size),
                        scale=tuple(rrc["scale"]),
                        ratio=tuple(rrc["ratio"])
                    )],
                    p=rrc.get("p", 0.5)
                ))
            if "JPEGCompression" in aug_cfg:
                jc = aug_cfg["JPEGCompression"]
                augmentation_transforms.append(transforms.RandomApply(
                    [v2.JPEG(
                        tuple(jc["quality_range"])
                    )],
                    p=jc.get("p", 0.5)
                ))

        # Base transformations including augmentations
        self.transform = transforms.Compose(augmentation_transforms + [
            transforms.Resize((48, 64)),
            transforms.ToTensor()   
        ])

        self.save_plots = save_plots
        self.plot_dir = os.path.join(dataset_dir, 'stat_plots')

        if self.save_plots and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Calculate Statistics
        print('Calculating Dataset Statistics...')
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
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
