import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import os
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import label2rgb
import matplotlib.pyplot as plt

data = []
input_image_folder=  '../synth_dataset/outputs_u_net/'
# filenames= os.listdir(input_image_folder)

# Load a binary image (example: black and white image with regions)
binary_image = imread('C:/Users/narges/PycharmProjects3/pythonProject3/IEEE_transaction_paper/real_dataset/outputs_light_u_net1/007_/binary_center.jpg', as_gray=True) > 0.5  # Threshold for binary

# Label connected regions
label_image = label(binary_image)

# Measure region properties
regions = regionprops(label_image)

# Display region properties
# for region in regions:
#     print(f"Region {region.label}:")
#     print(f"  Area: {region.area}")
#     print(f"  Centroid: {region.centroid}")
#     print(f"  Bounding Box: {region.bbox}")
print(len(regions))
# Optional visualization
plt.imshow(label2rgb(label_image, bg_label=0))
plt.title("Labeled Regions")
plt.show()
