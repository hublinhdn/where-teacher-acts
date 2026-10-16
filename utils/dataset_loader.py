import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class PillDataset(Dataset):
    # Extra argument: transform_ref
    def __init__(self, df, transform=None, transform_ref=None):
        self.df = df
        self.transform = transform
        # Without transform_ref, the shared transform is used for every image
        self.transform_ref = transform_ref if transform_ref is not None else transform
        self.root_path = 'data/raw/ePillID/classification_data'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_path, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = row['label_idx']
        sub_label = row['sub_label_idx']
        is_ref = row['is_ref']
        
        # ROUTE THE AUGMENTATION BY IMAGE DOMAIN
        if is_ref == 1 and self.transform_ref:
            image = self.transform_ref(image)
        elif self.transform:
            image = self.transform(image)

        # Return sub_label for training and the original label for evaluation
        return image, sub_label, label, is_ref

def get_transforms(is_train=True, size=448): # Default size raised to 448
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Rotation limited to 15 degrees so the imprint text is not turned upside down
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.label_to_indices = {l: np.where(labels == l)[0] for l in np.unique(labels)}
        self.labels_list = list(self.label_to_indices.keys())

    def __iter__(self):
        for _ in range(len(self.labels) // (self.n_classes * self.n_samples)):
            batch = []
            classes = np.random.choice(self.labels_list, self.n_classes, replace=False)
            for cls in classes:
                indices = self.label_to_indices[cls]
                batch.extend(np.random.choice(indices, self.n_samples, replace=True))
            yield batch

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)