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

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   
    # Create a window
    cv2.namedWindow('DoH Cell Counting')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Sigma', 'DoH Cell Counting', 1, 10, nothing)
    cv2.createTrackbar('Threshold', 'DoH Cell Counting', 10, 255, nothing)
    counter+=1

    while True:
        # Get current positions of trackbars
        sigma = cv2.getTrackbarPos('Sigma', 'DoH Cell Counting')
        threshold = cv2.getTrackbarPos('Threshold', 'DoH Cell Counting')
        
        # Apply Difference of Gaussian (DoG) to approximate Determinant of Hessian (DoH)
        dog = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma) - cv2.GaussianBlur(image, (0, 0), sigmaX=sigma*2)
        
        # Apply Threshold
        _, binary = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
        
        # Count the number of cells using connected components
        num_labels, labels_im = cv2.connectedComponents(binary.astype(np.uint8))
        error = np.abs(num_labels - 1- int(contents))/int(contents)

        # Display the number of cells on the image
        image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_display, f'Cells: {num_labels - 1}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_display, f'gt counts: {contents}', (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_display, f'error: {error*100}', (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show the binary result and the original image with cell count
        cv2.imshow('DoH Cell Counting', image_display)
        cv2.imshow('Binary', binary)
        
        # # Exit when 'q' key is pressed
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Move to the next image
            break
        # Destroy all windows
    data.append([file_name , contents, num_labels - 1 , error *100])

    cv2.destroyAllWindows()
    counter += 1

df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'DOH' , 'Error' ])
df.to_csv('../real_dataset_results/DoH_Light_U_Net1.csv', index=False)

