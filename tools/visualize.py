import os
import pydicom
from pydicom import dcmread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

path = sys.argv[1]
image_paths = os.listdir(path)
image_paths = sorted([os.path.join(path,i) for i in image_paths if i.endswith(".DCM")])
fig, ax = plt.subplots()

imgs = []
for im in image_paths:
    img = dcmread(im, force=True)
    img.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    data = img.pixel_array
    imgs.append([ax.imshow(data, animated=True)])

ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True)
plt.show()
