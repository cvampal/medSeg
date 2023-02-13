import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import pydicom
from tqdm import tqdm

class CTScanDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        super(CTScanDataset).__init__()
        self.transforms = transforms
        self.class_names = ["patient","PTV","CTV_P","CTV_FF","CTV_NV","GTV_P"] 
        self.data_dir = data_dir
        self.images_names = sorted(os.listdir(os.path.join(self.data_dir, "images")))
        self.masks = self._get_mask()

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir,"images", self.images_names[idx])
        img = pydicom.dcmread(img_path, force=True)
        img.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = img.pixel_array
        mask = torch.from_numpy(self.masks[:,:,idx]).to(torch.int64) 
        if self.transforms:
            img = self.transforms(img)
        return img, mask


    def _get_mask(self, size=(512,512,285)):
        mask = np.zeros(size,dtype=np.int64)
        i = 1
        for cl in self.class_names:
            array = np.load(os.path.join(self.data_dir, "segmentation-masks"  ,cl+".npy"))
            mask[array==1] = i # assuming that all binary mask are disjoints. otherwise incorrect.
            i+=1
        return mask



class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        return image, target


class Normalize:
    def __call__(self, image):
        mean = image[image.ne(0.0)].mean()
        std = image[image.ne(0.0)].std()
        if (mean.isnan().item()  or std.isnan().item()):
            return image
        else:
            return (image-mean)/std


class CTScanDatasetModule(pl.LightningDataModule):

    def setup(self, stage):
        transform=transforms.Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float32), Normalize()])

        ds = CTScanDataset("/home/shraddha/data/CT-AIIMS-Bhopal/", transforms=transform)
        self.train, self.val = torch.utils.data.random_split(ds,[0.75, 0.25])


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8, num_workers=4)






