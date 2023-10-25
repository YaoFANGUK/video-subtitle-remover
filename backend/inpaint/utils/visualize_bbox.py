import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


img = plt.imread('../example/fill-anything/sample1.png')
fig, ax = plt.subplots(1)
ax.imshow(img)

x1, y1, x2, y2 = 230, 283, 352, 407
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.savefig('bbox.png')
