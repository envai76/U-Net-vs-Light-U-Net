import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import os


def calculate_mae(predicted_density_map, ground_truth_density_map):
    """
    Calculate the Mean Absolute Error (MAE) between the predicted and ground truth density maps.

    Parameters:
    - predicted_density_map (numpy.ndarray): Predicted density map of shape (N, H, W, C).
    - ground_truth_density_map (numpy.ndarray): Ground truth density map of shape (N, H, W, C).

    Returns:
    - float: The calculated MAE across all spatial locations and all images.
    """
    # Ensure the input arrays have the same shape
    assert predicted_density_map.shape == ground_truth_density_map.shape, \
        "Predicted and ground truth density maps must have the same shape."

    # Compute the absolute differences
    absolute_differences = np.abs(predicted_density_map - ground_truth_density_map)

    # Compute the mean absolute error (MAE)
    mae = np.mean(absolute_differences)

    return mae

columns = ['filename1', 'synth u net','filename2',  'synth light u net' ,'filename3', 'real u net'  ,'filename4', 'real light u net']
df = pd.DataFrame(columns=columns)
input_image_folders= ['../synth_dataset/outputs_u_net/',   '../synth_dataset/outputs_light_u_net/','../real_dataset/outputs_u_net/', '../real_dataset/outputs_light_u_net1/']


for col_index in range(0, len(input_image_folders) ,1 ):
    folder= input_image_folders[col_index]
    filenames= os.listdir(folder)

    column_name = columns[col_index*2]  # Name of files
    column_name1 = columns[col_index*2+1]  # values of mae
    results = [] 
    filenames_col=[]
    print("folder name",folder)
    print(column_name , column_name1)
    i=0
    for file_name in filenames :

        # Load the original image (replace 'image_path' with the actual path to your image)
        predicted_density_map   = folder + str(file_name) + '/image_pr_centroids.jpg'
        ground_truth_density_map   = folder + str(file_name) + '/image_gt_centroids.jpg'
        
        predicted_density_map_im = io.imread(predicted_density_map, as_gray=True)
        
        ground_truth_density_map_im = io.imread(ground_truth_density_map, as_gray=True)
        
        mae = calculate_mae(predicted_density_map_im, ground_truth_density_map_im)
        # print("*"*10)
        # print(f'File_Name : {file_name}')
        # print("Mean Absolute Error (MAE):", mae)
        # data.append( [file_name , mae])
        results.append(mae)
        filenames_col.append(file_name)
        i=i+1
    if i <80:
        for i in range(80-i):
            results.append(0)
            filenames_col.append('-')
    
    df[column_name] = pd.Series(filenames_col)
    df[column_name1] = pd.Series(results)
    



# df = pd.DataFrame(data, columns=['Image name','synth u net', 'synth light u net' , 'real u net'  , 'real light u net' ])
df.to_csv('./density_maps_comp1.csv', index=False)
