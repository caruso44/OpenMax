from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class Satelite_images(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,self.images[idx])
        mask_path = os.path.join(self.mask_dir,self.images[idx].replace(".jpg","_mask.gif"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255] = 1
        augmentation = self.transform(image=image,mask=mask)
        image = augmentation["image"]
        mask = augmentation["mask"]
        

        return image,mask