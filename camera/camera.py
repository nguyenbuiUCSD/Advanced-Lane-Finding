import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import pickle

class Camera:
    def __init__(self, name, config_folder = "camera_config"):
        
        print ("Initializing camera ...")
        self.name = name
        self.is_calibrated = False
        config_filename = name+".conf"
        if config_filename in os.listdir(config_folder):
            print ("Found config file ...")
            print ("Loading configuration for camera ...")
            # Read in the saved objpoints and imgpoints
            dist_pickle = pickle.load( open( config_folder+"/"+config_filename, "rb" ) )
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]
            self.is_calibrated = True
        else:
            print ("Camera has not been calibrated yet ...")
            
        print ("Done ...")
    
    def find_chessboard_corners(self, image, nx, ny):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        return cv2.findChessboardCorners(gray, (nx, ny), None)
    
    def find_chessboard_corners_and_draw(self, image, nx, ny):
        image_copy = image.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(image_copy, (nx, ny), corners, ret)
        return ret, corners, image_copy
    
    def calibrate_and_draw(self, calib_images_folder, nx, ny):
        """Calibrate camera using images from calid_images_folder"""
        
        # +-----------------+
        # | FINDING CORNERS |
        # +-----------------+
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        print ("Finding corners ... ")        
        for image_link in os.listdir(calib_images_folder):
            image = mpimg.imread(calib_images_folder+image_link)
            ret, corners = self.find_chessboard_corners(image, nx, ny)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        print ("Done ...")
        
        # +------------------+
        # | CALIBRATE CAMERA |
        # +------------------+
        print ("Calibrating Camera ... ")
        image_link = random.choice(os.listdir(calib_images_folder))
        image = mpimg.imread(calib_images_folder+image_link)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.is_calibrated = True
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "camera_config/"+self.name+".conf", "wb" ) )
        print ("Camera configuration has been written to " + "camera_config/" + self.name+".conf")
        print ("Done ...")
        
        # +---------------+
        # | VISUALIZATION |
        # +---------------+
        # Finding corners
        print ("Showing random sample")
        image_link = random.choice(os.listdir(calib_images_folder))
        image1 = mpimg.imread(calib_images_folder+image_link)
        num_of_try = 10
        ret, corners, image_draw1 = self.find_chessboard_corners_and_draw(image1, nx, ny)
        while ret==False and num_of_try>0:
            image_link = random.choice(os.listdir(calib_images_folder))
            image1 = mpimg.imread(calib_images_folder+image_link)
            ret, corners, image_draw1 = self.find_chessboard_corners_and_draw(image, nx, ny)
            num_of_try-=1
        # Undistortion
        image_link = random.choice(os.listdir(calib_images_folder))
        image2 = mpimg.imread(calib_images_folder+image_link)
        image_draw2 = cv2.undistort(image2, mtx, dist, None, mtx)
        #Plot
        f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(20,10))
        ax11.imshow(image1)
        ax11.set_title('Original Image', fontsize=30)
        ax12.imshow(image_draw1)
        ax12.set_title('Corners Image', fontsize=30)          
        ax21.imshow(image2)
        ax21.set_title('Distorted Image', fontsize=30)
        ax22.imshow(image_draw2)
        ax22.set_title('Undistorted Image', fontsize=30)
        plt.show()
        
        
    def calibrate(self, calib_images_folder, nx, ny):
        """Calibrate camera using images from calid_images_folder"""

        # +-----------------+
        # | FINDING CORNERS |
        # +-----------------+
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        print ("Finding corners ... ")        
        for image_link in os.listdir(calib_images_folder):
            image = mpimg.imread(calib_images_folder+image_link)
            ret, corners = self.find_chessboard_corners(image, nx, ny)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        print ("Done ...")

        # +------------------+
        # | CALIBRATE CAMERA |
        # +------------------+
        print ("Calibrating Camera ... ")
        image_link = random.choice(os.listdir(calib_images_folder))
        image = mpimg.imread(calib_images_folder+image_link)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.is_calibrated = True
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "camera_config/"+self.name+".conf", "wb" ) )
        print ("Camera configuration has been written to " + "camera_config/" + self.name+".conf")
        print ("Done ...")

    def undistort(self, image):
        image_undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return image_undistorted