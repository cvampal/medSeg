import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import pydicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate3D(array3d):
    fig, ax = plt.subplots()
    imgs = []
    for i in range(array3d.shape[-1]):
        im = ax.imshow(array3d[:,:,i], animated=True)
        imgs.append([im])
    ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True)
    plt.show()
    
def get_binary_mask(dicom_series_path, rt_struct_path):
    out = {}
    rtstruct = RTStructBuilder.create_from(dicom_series_path, rt_struct_path)
    for arr in rtstruct.get_roi_names():
        mask_3d = rtstruct.get_roi_mask_by_name(arr)
        out[arr] = mask_3d 
    return out

check_overlap = lambda x1, x2 : (x1==1)[x2==1].sum()

def check_overlap_in_mask(binary_masks):
    overlap = {}
    for i in range(len(binary_masks)):
        for j in range(len(binary_masks)):
            if i>j:
                overlap[f"{i}-{j}"] = check_overlap(binary_masks[i], binary_masks[j])
    return overlap

if __name__ == '__main__':
    dicom_series_path="../data/CT-AIIMS-Bhopal/" 
    rt_struct_path="../data/CT-AIIMS-Bhopal/0390222_StrctrSets.dcm"
    ms = get_binary_mask(dicom_series_path, rt_struct_path)
    print(ms.keys())
    print(check_overlap_in_mask(list(ms.values())))
    #for k in ms.keys():
    #    np.save(f"../data/{k}.npy", ms[k])





