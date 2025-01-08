import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, feature, color
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd
import os

def detect_cells(density_map, sigma=2, threshold_factor=1.5):
    """
    Detect cells using local maxima detection on a smoothed density map.

    Parameters:
        density_map (numpy.ndarray): Input density map (2D array).
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold_factor (float): Factor to determine threshold for local maxima based on mean and std.

    Returns:
        cell_centroids (list of tuples): List of detected cell centroids as (x, y) coordinates.
    """
    # Smooth the density map using a Gaussian filter
    smoothed_map = gaussian_filter(density_map, sigma=sigma)

    # Calculate threshold based on the mean and standard deviation of the smoothed map
    mean_val = np.mean(smoothed_map)
    std_val = np.std(smoothed_map)
    threshold = mean_val + threshold_factor * std_val

    # Identify local maxima above the threshold
    coordinates = peak_local_max(smoothed_map, min_distance=3, threshold_abs=threshold)

    # Convert to a list of (x, y) tuples
    cell_centroids = [(int(x), int(y)) for y, x in coordinates]

    return cell_centroids


data =[]
# Example usage
input_image_folder=  '../real_dataset/outputs_light_u_net1/'

filenames= os.listdir(input_image_folder)
for file_name in filenames :
    # Load or generate a sample density map (replace with your own data)
    # For demonstration, we'll create a synthetic density map
    image_path   = input_image_folder + str(file_name) + '/image_pr_centroids.jpg'
    real_counts  = input_image_folder + str(file_name) + '/counts.txt'
    file_path    = input_image_folder + str(file_name) + '/predicted_counts.txt'
    address= input_image_folder + str(file_name) + '/local_maximas.jpg'
    density_map = io.imread(image_path, as_gray=True)
    with open(real_counts, 'r') as file:
        contents = file.read()

    # Detect cells
    cell_centroids = detect_cells(density_map, sigma=2, threshold_factor=1.5)
   
    print("*"*10)

    print("image : " , file_name )
    print( f"Local Maxima: {len(cell_centroids)}")  # Subtract 1 to ignore the background component
    print(f"error {np.abs(len(cell_centroids) - 1 -int(contents) )/ int(contents) *100}"  )
    data.append([file_name , contents, len(cell_centroids) , np.abs(len(cell_centroids) -int(contents) )/ int(contents) *100])

    with open(file_path, 'a') as file:
        file.write("Local Maxima: "+ str(len(cell_centroids)) + '\n')


    # Visualize the results

    plt.figure(figsize=(8, 4))

    # # Original density map
    # plt.subplot(1, 2, 1)
    # plt.title("Density Map")
    # plt.imshow(density_map, cmap="hot")
    # plt.colorbar()

    # # Smoothed density map with detected centroids
    # plt.subplot(1, 2, 2)
    # plt.title("Detected Centroids")
    plt.imshow(density_map, cmap="hot")
    for x, y in cell_centroids:
        plt.plot(x, y, "bo")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()
    plt.savefig(address)

  

    

df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'Local Maxima' , 'Error' ])
df.to_csv('real_dataset_results/Local_Maxima_Light_U_Net1.csv', index=False)
