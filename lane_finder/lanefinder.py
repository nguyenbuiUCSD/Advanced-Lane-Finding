import numpy as np
import cv2

class Lanefinder:
    def __init__(self):
        print ("Initializing image processor ..")
        print ("Default value of sliding window have been set")
        print ("window_width = 50") 
        print ("window_height = 80")
        print ("margin = 100")
        self.window_width = 50 
        self.window_height = 80
        self.margin = 100
        print ("Done ...")
        
     
    def set_sliding_windows(self, window_width = 50, window_height = 80, margin = 100):
        self.window_width = window_width 
        self.window_height = window_height
        self.margin = margin
    
    def find_lane_segments(self, image):
        # return value
        window_centroids = []
        # Kernel
        window = np.ones(self.window_width)
        # Use left side of window as center
        offset = int(self.window_width/2)
        
        #Start from bottom
        # Calculate histogram of bottom 1/4 sliced image for initial position
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-offset
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-offset+int(image.shape[1]/2)
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(0,(int)(image.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*self.window_height):\
                                       int(image.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
        return window_centroids