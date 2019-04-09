import numpy as np
import matplotlib.pyplot as plt
import cv2

class Lanefinder:
    def __init__(self):
        print ("Initializing image processor ..")
        print ("Default value of sliding window have been set")
        print ("window_width = 50") 
        print ("window_height = 80")
        print ("margin = 100")
        print ("minimum peak for centroid = 15")
        self.window_width = 50 
        self.window_height = 80
        self.margin = 100
        self.min_peak = 15
        print ("Done ...")
        
     
    def set_sliding_windows(self, window_width = 50, window_height = 80, margin = 100):
        self.window_width = window_width 
        self.window_height = window_height
        self.margin = margin
    
    def find_lane_segments(self, image):
        # return value
        left_centroids = []
        right_centroids = []
        # Kernel
        window = np.ones(self.window_width)
        conv_mode = "same"
        # Offset: When finding centroids, the centroid is the starting point on the right side
        #         of the lane mark
        offset = 10
        
        
        #Start from bottom
        # Calculate histogram of bottom 1/4 sliced image for initial position
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum,mode=conv_mode))
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum,mode=conv_mode))+int(image.shape[1]/2)
        
        # Go through each layer looking for max pixel locations
        for level in range(0,(int)(image.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*self.window_height):\
                                       int(image.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer,mode=conv_mode)
            # Find the best left centroid by using past left center as a reference
            l_min_index = int(max(l_center-self.margin,0))
            l_max_index = int(min(l_center+self.margin,image.shape[1]))
            l_center_temp = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index
            if conv_signal[l_center_temp]>self.min_peak:
                l_center = l_center_temp
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center-self.margin,0))
            r_max_index = int(min(r_center+self.margin,image.shape[1]))
            r_center_temp = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index
            if conv_signal[r_center_temp]>self.min_peak:
                r_center = r_center_temp
            # Add what we found for that layer
            y_center = int(image.shape[0]-level*self.window_height)
            left_centroids.append((l_center+offset,y_center))
            right_centroids.append((r_center+offset,y_center))
        return left_centroids, right_centroids
    
    
    def fit_curve(self, pts):
        pts_array = np.array(pts)
        poly_fit = np.polyfit(pts_array[:,1], pts_array[:,0], 2)
        y_eval = np.max(pts_array[:,1])
        curverad = ((1 + (2*poly_fit[0]*y_eval + poly_fit[1])**2)**1.5) / np.absolute(2*poly_fit[0])
        return poly_fit, curverad
    
    
    