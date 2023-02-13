#!/usr/bin/env python
# coding: utf-8

# ## Generating 2D images and masks from 3D MRI images given slice range:

import nibabel
import numpy as np
import os
import sys



from tqdm import tqdm


import random

unzip = lambda x : ([i for i,_ in x],[i for _,i in x])

def random_split(A,B,pro=0.75):
    l = list(zip(A,B))
    random.shuffle(l)
    by = int(len(l)*pro)
    return unzip( l[:by]), unzip(l[by:])


def create_mri2d_data(root,path_to_save,slice_range=(60,110)):
    images_list = list(sorted( f for f in os.listdir(os.path.join(root, "imagesTr")) if not f.startswith(".")) )
    labels_list = list(sorted( f for f in os.listdir(os.path.join(root, "labelsTr")) if not f.startswith(".")) )
    (img_tr, mask_tr), (img_val, mask_val) = random_split(images_list,labels_list)
    generate(root, path_to_save,img_tr, mask_tr, slice_range=slice_range, mode='train')
    generate(root, path_to_save,img_val, mask_val, slice_range=slice_range, mode='val')
    print("done")


def generate(root,path_to_save,images_list, labels_list, slice_range=(60,110),mode="train"):
    for i in tqdm(range(len(images_list))):
        image = nibabel.load(os.path.join(root,"imagesTr",images_list[i])).get_fdata()
        mask  = nibabel.load(os.path.join(root,"labelsTr",labels_list[i])).get_fdata()
        for m in range(4):
            for idx in range(*slice_range):
                im = image[:,:,idx,m].astype(int)
                msk = mask[:,:,idx].astype(int)
                np.save(f"{path_to_save}/images/{mode}/{m}-{idx}-{i}.npy", im)
                np.save(f"{path_to_save}/annotations/{mode}/{m}-{idx}-{i}.npy", msk)




if __name__ == '__main__':
    path = sys.argv[1]
    save = sys.argv[2]
    create_mri2d_data(path,save)




