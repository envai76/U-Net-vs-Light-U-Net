import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import os






columns = ['filename1', 'real count1', 'synth u net counts','error1', 
           'filename2', 'real count2', 'synth light u net counts','error2' ,
           'filename3','real count3', 'real u net counts','error3'  ,
           'filename4','real count4', 'real light u net counts' , 'error4']


df = pd.DataFrame(columns=columns)
input_image_folders= ['../synth_dataset/outputs_u_net/',   '../synth_dataset/outputs_light_u_net1/','../real_dataset/outputs_u_net/', '../real_dataset/outputs_light_u_net1/']


for col_index in range(0, len(input_image_folders) ,1 ):
    folder= input_image_folders[col_index]
    filenames= os.listdir(folder)

    column_name = columns[col_index*4]  # Name of files
    col_real_counts = columns[col_index*4+1]
    col_method_counts=columns[col_index*4+2]
    col_error = columns[col_index*4+3]

    errors = [] 
    filenames_col=[]
    method_count=[]
    real_counts_col=[]
    print("folder name",folder , column_name )
    i=0
    
    for file_name in filenames :


        # Load the original image (replace 'image_path' with the actual path to your image)
        image_path   = folder + str(file_name) + '/image_pr_regions.jpg'
        file_path    = folder + str(file_name) + '/predicted_counts.txt'
        real_counts  = folder + str(file_name) + '/counts.txt'
        image = io.imread(image_path, as_gray=True)
        with open(real_counts, 'r') as file:
            contents = file.read()
            print(contents)
        centroid_path = folder + str(file_name) + '/image_pr_centroids.jpg'
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
        error= np.abs(num_cells  -int(contents) )/ int(contents) *100
        # data.append( [file_name , contents, num_cells, np.abs(num_cells  -int(contents) )/ int(contents) *100])
        # data.append([file_name, num_cells])

        method_count.append(num_cells)
        real_counts_col.append(contents)
        filenames_col.append(file_name)
        errors.append(error)



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

        folder_name= folder +file_name
        output_path = os.path.join(folder_name, 'segments_incolour.jpg')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0) 

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        i= i+1

    if i <80:
        for i in range(80-i):
            filenames_col.append('')
            method_count.append('')
            real_counts_col.append('')
            errors.append('')
    
    
    df[column_name] = pd.Series(filenames_col)
    df[col_real_counts] = pd.Series(real_counts_col)
    df[col_method_counts] = pd.Series(method_count)
    df[col_error] = pd.Series(errors)
    



# df = pd.DataFrame(data, columns=['Image name','synth u net', 'synth light u net' , 'real u net'  , 'real light u net' ])
df.to_csv('./Watershed1.csv', index=False)