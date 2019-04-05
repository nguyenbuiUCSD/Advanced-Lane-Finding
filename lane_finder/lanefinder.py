import cv2
import numpy as np

class Lanefinder:
    def __init__(self):
        print ("Initializing lane finder ..")
        
        print ("Done ...")
        
    def abs_sobel_threshold(self, image, orient='x', sobel_kernel=3, sobel_threshold=(0, 255)):

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply inclusive (>=, <=) thresholds
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 1

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
    
    def hls_color_threshold(self, image, h_threshold=(256, 256), l_threshold=(256, 256), s_threshold=(256, 256)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_output = np.zeros_like(image[:,:,0])
        binary_output[(hls[:,:,0] >= h_threshold[0]) & (hls[:,:,0] <= h_threshold[1])] = 1
        binary_output[(hls[:,:,1] >= l_threshold[0]) & (hls[:,:,1] <= l_threshold[1])] = 1
        binary_output[(hls[:,:,2] >= s_threshold[0]) & (hls[:,:,2] <= s_threshold[1])] = 1
        return binary_output
    
    def combined_threshold(self, gradx, grady, mag_binary, dir_binary):
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined