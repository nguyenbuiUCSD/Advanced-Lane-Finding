## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"  

---

### Camera Calibration

#### 1. Computed the camera matrix and distortion coefficients.
```python
def calibrate(self, calib_images_folder, nx, ny):
        """Calibrate camera using images from calid_images_folder"""
```
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

```python
ret, corners = self.find_chessboard_corners(image, nx, ny)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
```
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
I applied this distortion correction to the test image using the `camera.undistort()` function and obtained this result: 

![camera1](https://user-images.githubusercontent.com/17399214/55820490-14ba2c00-5ab0-11e9-8154-529b26208fb7.png)

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Color transforms, gradients and other thresholds

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.

![undistorted](https://user-images.githubusercontent.com/17399214/55820658-78dcf000-5ab0-11e9-9a2e-d354b4fba897.png)


#### 3. Perspective transform

The code for my perspective transform includes a function called `set_perspective_transform()`, in imageprocessor class.  The `set_perspective_transform()` function takes as inputs source (`src`) and destination (`dst`) points. Then calculate the transformation matrix.

This following are source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580,458       | 400,20        | 
| 703,458       | 890,20        |
| 252, 678      | 400, 700      |
| 1054,678      | 890,700       |

```python
imageprocessor = Imageprocessor()
# Set perspective transform
image_top_left = (580,458)
image_top_right = (703,458)
image_bottom_left = (252, 678)
image_bottom_right = (1054,678)

birdeye_top_left = (400,20)
birdeye_top_right = (890,20)
birdeye_bottom_left = (400, 700)
birdeye_bottom_right = (890,700)

src = np.float32([image_top_left,image_bottom_left,image_top_right,image_bottom_right])
dst = np.float32([birdeye_top_left,birdeye_bottom_left,birdeye_top_right,birdeye_bottom_right])
imageprocessor.set_perspective_transform(src,dst)
```
![undistorted](https://user-images.githubusercontent.com/17399214/55820658-78dcf000-5ab0-11e9-9a2e-d354b4fba897.png)

#### 4. Find lane-line pixels and fit their positions with a polynomial?
Using convolutional algorithm to mark the hot pixel in binary image
![fitting](https://user-images.githubusercontent.com/17399214/55821717-f7d32800-5ab2-11e9-933f-d83fa469c9bb.png)

#### 5. Radius of curvature and the position of the vehicle with respect to center.
Radius and cervature are calculated in this fucntion:
```python
def fit_curve(self, pts):
        pts_array = np.array(pts)
        poly_fit = np.polyfit(pts_array[:,1], pts_array[:,0], 2)
        y_eval = np.max(pts_array[:,1])
        curverad = ((1 + (2*poly_fit[0]*y_eval + poly_fit[1])**2)**1.5) / np.absolute(2*poly_fit[0])
        return poly_fit, curverad
```
![curve](https://user-images.githubusercontent.com/17399214/55821821-32d55b80-5ab3-11e9-8f66-b34cad35cab0.png)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![result](https://user-images.githubusercontent.com/17399214/55821346-2270b100-5ab2-11e9-9122-f2eb76a06c38.png)


---

### Pipeline (video)

Here's a [link to my video result](https://github.com/nguyenbuiUCSD/Advanced-Lane-Finding/blob/master/test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Problems / issues
- Performance: this algorithm run quite slow.
- Turnable parametters: Al parametters are hard to twist. A set of parametters will not work on different road conditions
#### 2. Future improvement
- Using AI for lane mark segmentation
- Utilize paralel computing to improve performance
