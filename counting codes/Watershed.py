import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import os
data = []
input_image_folder=  '../synth_dataset/outputs_u_net/'
filenames= os.listdir(input_image_folder)
for file_name in filenames :


        # Load the original image (replace 'image_path' with the actual path to your image)
        image_path   = input_image_folder + str(file_name) + '/image_pr_regions.jpg'
        file_path    = input_image_folder + str(file_name) + '/predicted_counts.txt'
        real_counts  = input_image_folder + str(file_name) + '/counts.txt'
        image = io.imread(image_path, as_gray=True)
        with open(real_counts, 'r') as file:
            contents = file.read()
            print(contents)
        centroid_path = input_image_folder + str(file_name) + '/image_pr_centroids.jpg'
        centroid_image = io.imread(centroid_path, as_gray=True)

        # Apply Gaussian filter to smooth the original image (adjust sigma as needed)
        smoothed_image = filters.gaussian(centroid_image, sigma=1.5)

        # Segment the yeast cell regions using a threshold
        threshold = filters.threshold_otsu(smoothed_image)
        centroid_image = smoothed_image > threshold

        # Apply Gaussian filter to smooth the original image (adjust sigma as needed)
        smoothed_image = filters.gaussian(image, sigma=0.0001)

        # Segment the yeast cell regions using a threshold
        threshold = filters.threshold_otsu(smoothed_image)
        binary_image = smoothed_image > threshold

        # Compute distance transform using scipy.ndimage
        distance_transform = ndi.distance_transform_edt(binary_image)

        # Use the centroid image as markers for watershed segmentation
        markers = measure.label(centroid_image)

        # Apply watershed algorithm
        labels = watershed(-distance_transform, markers, mask=binary_image)

        # Remove small objects (adjust min_size as needed)
        labels = morphology.remove_small_objects(labels, min_size=50)
        # Output the number of labeled yeast cells
        num_cells = labels.max()
        print("*"*10)
        print(f'File_Name : {file_name}')
        print(f'counted: {num_cells}')
        print(f"error {np.abs(num_cells  -int(contents) )/ int(contents) *100}"  )
        data.append( [file_name , contents, num_cells, np.abs(num_cells  -int(contents) )/ int(contents) *100])
        # data.append([file_name, num_cells])

        with open(file_path, 'w') as file:
            file.write("Watershed: "+ str(num_cells) + '\n')

        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')

        axes[1].imshow(centroid_image, cmap='gray')
        axes[1].set_title('Centroid Image')
       
        axes[2].imshow(color.label2rgb(labels, image=image, bg_label=0))
        axes[2].set_title('Segmented Image with Watershed')

        folder_name= input_image_folder +file_name
        output_path = os.path.join(folder_name, 'segments_incolour.jpg')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0) 

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        # plt.show()

df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'Watershed' , 'Error' ])
df.to_csv('./synth_dataset_results/Watershed_U_Net.csv', index=False)
