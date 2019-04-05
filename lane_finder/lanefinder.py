import cv2
import numpy as np

class Lanefinder:
    def __init__(self):
        print ("Initializing lane finder ..")
        
        print ("Done ...")
        
    def abs_sobel_threshold(self, image, orient='x', threshold_min=0, threshold_max=255):

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply inclusive (>=, <=) thresholds
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= threshold_min) & (scaled_sobel <= threshold_max)] = 1

        # Return the result
        return binary_output
    
    def magnitude_threshold(self, image, sobel_kernel=3, magnitude_threshold=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(magnitude)/255 
        magnitude = (magnitude/scale_factor).astype(np.uint8) 
        # Create a copy and apply inclusive (>=, <=) thresholds
        binary_output = np.zeros_like(magnitude)
        binary_output[(magnitude >= magnitude_threshold[0]) & (magnitude <= magnitude_threshold[1])] = 1

        # Return the binary image
        return binary_output
    
    def direction_threshold(self, image, sobel_kernel=3, direction_threshold=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate direction
        abs_gradient_direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # Create a copy and apply inclusive (>=, <=) thresholds
        binary_output = np.zeros_like(abs_gradient_direction)
        binary_output[(abs_gradient_direction>=direction_threshold[0])&(abs_gradient_direction<=direction_threshold[1])] = 1

        # Return the binary image
        return binary_output