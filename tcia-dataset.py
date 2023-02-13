import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import nrrd
from tqdm import tqdm

class CTScanDataset(Dataset):
    def __init__(self, data_dir, mode, transforms=None):
        super(CTScanDataset).__init__()
        self.transforms = transforms
        self.class_names = ['Orbit_Lt', 'Lens_Rt', 'Lung_Lt', 'Brainstem', 'Parotid_Lt', 'Submandibular_Lt', 'Cochlea_Rt', 'Lacrimal_Lt', 'Optic_Nerve_Rt', 'Lung_Rt', 'Parotid_Rt', 'Brain', 'Mandible', 'Spinal_Canal', 'Lens_Lt', 'Lacrimal_Rt', 'Submandibular_Rt', 'Orbit_Rt', 'Spinal_Cord', 'Cochlea_Lt', 'Optic_Nerve_Lt']
        if mode == "train":
            self.data_dir = os.path.join(data_dir,"nrrds/test/oncologist")
        if mode == "val":
            self.data_dir = os.path.join(data_dir,"nrrds/validation/oncologist")
            
        self.images_names = os.listdir(self.data_dir)
        self.xs = []
        self.ys = []
        self._create_dataset()

    def _create_dataset(self):
        for idx in tqdm(range(len(self.images_names))):
            i,m = self._get_img(idx)
            for j in range(i.shape[-1]):
                i2d = i[:,:,j]
                m2d = m[:,:,j]
                if self.transforms:
                    i2d = self.transforms(i2d)
                    m2d = torch.from_numpy(m2d).to(dtype=torch.int64)
                self.xs.append(i2d)
                self.ys.append(m2d)



    def _get_img(self, idx):
        img_path = os.path.join(self.data_dir, self.images_names[idx])
        img, _ = nrrd.read(f"{img_path}/CT_IMAGE.nrrd")
        mask  = self._get_mask(img_path, img.shape)
        return img, mask

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


    def _get_mask(self, path, size):
        mask = np.zeros(size, dtype=np.int64)
        s = size[2]
        for i,cl in enumerate(self.class_names):
            array, _ = nrrd.read(os.path.join(path, "segmentations", cl+".nrrd"))
            diff = s - array.shape[2]
            if diff > 0:
                padd = np.zeros((size[0],size[1],diff), dtype=np.uint8)
                array = np.concatenate((array, padd), axis=2)
            mask[array == 1] = i+1
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

        self.CTScan_train = CTScanDataset("../data/tcia-ct-scan-dataset/", mode="train", transforms=transform)
        self.CTScan_val = CTScanDataset("../data/tcia-ct-scan-dataset/", mode="val", transforms=transform)


    def train_dataloader(self):
        return DataLoader(self.CTScan_train, batch_size=16, num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.CTScan_val, batch_size=8, num_workers=4)







