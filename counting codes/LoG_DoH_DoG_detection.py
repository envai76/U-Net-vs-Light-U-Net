import cv2
import numpy as np
import os
import pandas as pd
# Callback function for trackbar (it does nothing but is required)
def nothing(x):
    pass

data =[]
# Example usage
input_image_folder=  '../real_dataset/outputs_u_net/'
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

    if image is None:
        print("Error: Image not found!")
        exit()

    # Normalize the image for consistent processing
    normalized_image = cv2.normalize(image.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

    # Create windows and trackbars for LoG, DoG, and DoH
    cv2.namedWindow("LoG", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DoG", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DoH", cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Sigma', 'LoG', 1, 20, nothing)        # Sigma for LoG
    cv2.createTrackbar('Threshold', 'LoG', 10, 255, nothing) # Threshold for LoG

    cv2.createTrackbar('Sigma', 'DoG', 1, 20, nothing)        # Sigma for DoG
    cv2.createTrackbar('Threshold', 'DoG', 10, 255, nothing) # Threshold for DoG

    cv2.createTrackbar('Threshold', 'DoH', 10, 255, nothing) # Threshold for DoH

    while True:
        # Get trackbar positions
        sigma_log = cv2.getTrackbarPos('Sigma', 'LoG')
        sigma_dog = cv2.getTrackbarPos('Sigma', 'DoG')
        log_threshold = cv2.getTrackbarPos('Threshold', 'LoG')
        dog_threshold = cv2.getTrackbarPos('Threshold', 'DoG')
        doh_threshold = cv2.getTrackbarPos('Threshold', 'DoH')

        # Ensure sigma values are valid
        if sigma_log <= 0:
            sigma_log = 1
        if sigma_dog <= 0:
            sigma_dog = 1

        # Apply Laplacian of Gaussian (LoG)
        blurred_log = cv2.GaussianBlur(normalized_image, (0, 0), sigmaX=sigma_log)
        log_result = cv2.Laplacian(blurred_log, cv2.CV_64F)
        abs_log = np.abs(log_result)
        _, log_binary = cv2.threshold(abs_log, log_threshold / 255.0, 1, cv2.THRESH_BINARY)
        log_binary_display = (log_binary * 255).astype(np.uint8)
        num_labels_log, labels_im_log = cv2.connectedComponents(log_binary_display)

        # Apply Difference of Gaussian (DoG)
        dog = cv2.GaussianBlur(normalized_image, (0, 0), sigmaX=sigma_dog) - cv2.GaussianBlur(normalized_image, (0, 0), sigmaX=sigma_dog * 2)
        _, dog_binary = cv2.threshold(dog, dog_threshold / 255.0, 1, cv2.THRESH_BINARY)
        dog_binary_display = (dog_binary * 255).astype(np.uint8)
        num_labels_dog, labels_im_dog = cv2.connectedComponents(dog_binary_display)

        # Apply Determinant of Hessian (DoH)
        laplacian = cv2.Laplacian(normalized_image, cv2.CV_64F)
        abs_laplacian = np.abs(laplacian)
        _, doh_binary = cv2.threshold(abs_laplacian, doh_threshold / 255.0, 1, cv2.THRESH_BINARY)
        doh_binary_display = (doh_binary * 255).astype(np.uint8)
        num_labels_doh, labels_im_doh = cv2.connectedComponents(doh_binary_display)

        # Prepare display for LoG, DoG, and DoH
        log_display = cv2.normalize(log_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dog_display = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        doh_display = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Annotate blob counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(log_display, f'LoG Blobs: {num_labels_log - 1}', (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dog_display, f'DoG Blobs: {num_labels_dog - 1}', (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(doh_display, f'DoH Blobs: {num_labels_doh - 1}', (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)



        error_log = np.abs(num_labels_log - 1- int(contents))/int(contents)*100
        error_dog = np.abs(num_labels_dog - 1- int(contents))/int(contents)*100
        error_doh = np.abs(num_labels_doh - 1- int(contents))/int(contents)*100
        cv2.putText(log_display, f'gt: {contents}', (10, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dog_display, f'gt: {contents}', (10, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(doh_display, f'gt: {contents}', (10, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(log_display, f'error: {error_log}', (10, 200), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dog_display, f'error: {error_dog}', (10, 200), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(doh_display, f'error: {error_doh}', (10, 200), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        # Display results
        cv2.imshow('LoG', log_display)
        cv2.imshow('LoG Binary', log_binary_display)
        cv2.imshow('DoG', dog_display)
        cv2.imshow('DoG Binary', dog_binary_display)
        cv2.imshow('DoH', doh_display)
        cv2.imshow('DoH Binary', doh_binary_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows
    cv2.destroyAllWindows()
    data.append([file_name , contents, num_labels_log - 1 , error_log , num_labels_dog - 1 , error_dog , num_labels_doh - 1 , error_doh])


df = pd.DataFrame(data, columns=['Image name','Ground Truth', 'LoG' , 'Error', 'DoG' , 'Error', 'DoH' , 'Error' ])
df.to_csv('real_dataset_results/LoG_DoG_DoH_U_Net.csv', index=False)

