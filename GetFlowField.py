from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('Left.png')
raw_flow = np.array(im)

max = 255
min = 0
norm = raw_flow / max
# print(norm)
scaled = norm - 0.5

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.imshow(scaled[500:, :500, 0])
# ax2.imshow(scaled[500:, :500, 1])
# fig.tight_layout()


plt.figure()
plt.imshow(norm[500:, :500, 0])
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.imshow(norm[500:, :500, 1])
plt.colorbar()
plt.tight_layout()

plt.show()
