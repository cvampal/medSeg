import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nrrd
import sys

fig, ax = plt.subplots()

img, _ = nrrd.read(sys.argv[1])
img = np.array(img)


ims = []
for i in range(img.shape[-1]):
    im = ax.imshow(img[:,:,i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
