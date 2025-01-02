import cv2
import numpy as np
from skimage import io, filters, morphology, measure, feature, color
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import re
import os
counts=[]
data=[]
input_image_folder=  '../synth_dataset/outputs_u_net/'
filenames= os.listdir(input_image_folder)
for file_name in filenames :


    centroid_path   = input_image_folder + str(file_name) + '/image_pr_centroids.jpg'
    file_path    = input_image_folder + str(file_name) + '/predicted_counts.txt'
    real_counts  = input_image_folder + str(file_name) + '/counts.txt'

    address=        input_image_folder + str(file_name) + '/binary_center.jpg'
    address2=       input_image_folder + str(file_name) + '/cca_output.jpg'

    with open(real_counts, 'r') as file:
            contents = file.read()
            print(contents)
    centroid_image = io.imread(centroid_path, as_gray=True)

    

    # Apply Gaussian filter to smooth the original image (adjust sigma as needed)
    smoothed_image = filters.gaussian(centroid_image, sigma=1.5)

    # Segment the yeast cell regions using a threshold
    threshold = filters.threshold_otsu(smoothed_image)
    centroid_image = smoothed_image > threshold
   

    # Apply a binary threshold to the image to convert it to a binary image
    # Adjust the threshold value if necessary
    # _, binary_image = cv2.threshold(centroid_image, 127, 255, cv2.THRESH_BINARY)
    threshold = filters.threshold_otsu(smoothed_image)
    binary_image = smoothed_image > threshold
    binary_image_uint8 = (binary_image.astype(np.uint8)) * 255

    cv2.imwrite(address, binary_image_uint8)
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_image_uint8)
    # Print the number of connected components
    print("*"*10)

    print("image : " , file_name )
    print( f"Connected Components: {num_labels - 1}")  # Subtract 1 to ignore the background component
    print(f"error {np.abs(num_labels - 1 -int(contents) )/ int(contents) *100}"  )
    data.append([file_name , contents, num_labels - 1 , np.abs(num_labels - 1 -int(contents) )/ int(contents) *100])

    with open(file_path, 'a') as file:
        file.write("Connected Component: "+ str(num_labels - 1) + '\n')



    # Optionally, visualize the components
    # Create a color map to represent different components
    output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colors = []

    for i in range(1, num_labels):
        colors.append(np.random.randint(0, 255, size=3))

    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] > 0:
                output[y, x] = colors[labels[y, x] - 1]

    # Save or display the output image
    # cv2.imwrite('connected_components.png', output)
    # cv2.imshow('Connected Components', output)
    cv2.imwrite(address2 , output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'Connected Component' , 'Error' ])
df.to_csv('synth_dataset_results/Connected_Comp_U_Net.csv', index=False)
