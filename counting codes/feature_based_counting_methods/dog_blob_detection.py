import cv2
import numpy as np
import os
import pandas as pd
# Callback function for trackbar (it does nothing but is required)
def nothing(x):
    pass
data =[]
# Example usage
input_image_folder=  '../../real_dataset/outputs_light_u_net1/'
counter=0
filenames= os.listdir(input_image_folder)
for file_name in filenames :
    
    # Load or generate a sample density map (replace with your own data)
    # For demonstration, we'll create a synthetic density map
    image_path   = input_image_folder + str(file_name) + '/image_pr_centroids.jpg'
    real_counts  = input_image_folder + str(file_name) + '/counts.txt'
    # file_path    = input_image_folder + str(file_name) + '/predicted_counts.txt'
    # address= input_image_folder + str(file_name) + '/local_maximas.jpg'
    # density_map = io.imread(image_path, as_gray=True)
    with open(real_counts, 'r') as file:
        contents = file.read()

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # Create a window
    cv2.namedWindow('DoG Cell Counting')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Blur1', 'DoG Cell Counting', 5, 20, nothing)
    cv2.createTrackbar('Blur2', 'DoG Cell Counting', 9, 20, nothing)
    cv2.createTrackbar('Threshold', 'DoG Cell Counting', 20, 255, nothing)

    while True:
        # Get current positions of trackbars
        blur1_val = cv2.getTrackbarPos('Blur1', 'DoG Cell Counting')
        blur2_val = cv2.getTrackbarPos('Blur2', 'DoG Cell Counting')
        thresh_val = cv2.getTrackbarPos('Threshold', 'DoG Cell Counting')
        
        # Ensure blur_val is odd (GaussianBlur requires odd kernel size)
        if blur1_val % 2 == 0:
            blur1_val += 1
        if blur2_val % 2 == 0:
            blur2_val += 1

        # Apply Gaussian Blur
        blurred1 = cv2.GaussianBlur(gray, (blur1_val, blur1_val), 0)
        blurred2 = cv2.GaussianBlur(gray, (blur2_val, blur2_val), 0)
        
        # Compute Difference of Gaussians (DoG)
        dog = cv2.subtract(blurred1, blurred2)
        
        # Apply Threshold
        _, binary = cv2.threshold(dog, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Count the number of cells using connected components
        num_labels, labels_im = cv2.connectedComponents(binary)
        error = np.abs(num_labels - 1- int(contents))/int(contents)

        # Display the number of cells on the image
        image_display = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_display, f'Cells: {num_labels - 1}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_display, f'gt counts: {contents}', (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_display, f'error: {error*100}', (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show the binary result and the original image with cell count
        cv2.imshow('DoG Cell Counting', image_display)
        cv2.imshow('Binary', binary)
        
        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows
    cv2.destroyAllWindows()

df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'DOG' , 'Error' ])
df.to_csv('../real_dataset_results/DoG_Light_U_Net1.csv', index=False)

