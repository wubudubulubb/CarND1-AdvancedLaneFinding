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

[image1]: ./output_images/undistorted_calibration_3.jpg "Undistorted"
[image2]: ./cal_images/calibration_3.jpg "calibration image 3"
[image3]: ./test_images/straight_lines1.jpg "Straight Lines Original"
[image4]: ./output_images/undistorted_straight_lines1.jpg "Straight Lines 1 undistorted"
[image5]: ./output_images/color_example.png "Example for color filtering"
[image6]: ./output_images/warping.png "Warping"
[image7]: ./output_images/testing_the_pipeline.png "pipeline output for test5.jpg"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "AdvancedLaneFinding.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image using `cv2.findChessboardCorners()`.  `imgpoints` are be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image "calibration_3.jpg" using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Original Image :

![alt text][image3]

Undistorted :
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used only the color thresholds to generate a binary image (thresholding steps in the section of "Color Thresholding" in `AdvancedLaneFinding.ipynb`).  Here's an example of my output for this step.  

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `change_perspective()`, which appears in section "Perspective Transform" in the file `AdvancedLaneFinding.ipynb`.  The `change_perspective()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. It returns the warped image, transformation matrix and the inverse transformation matrix.  I hardcoded the source and destination points (which were found by trial and error by visual observation) in the following manner:

```python
width, height = 1280, 720

bottom_left = [260, 669]
top_left = [530, 492]
top_right = [760, 492]
bottom_right = [1040, 669]

left_offset = 320.0
right_offset = (width - bottom_right[0]) / np.float32(bottom_left[0]) * left_offset
top_offset = 450.0
bottom_offset = 1.0

bottom_left_dst = [left_offset, height - bottom_offset]
top_left_dst = [left_offset, top_offset]
top_right_dst = [width - right_offset, top_offset]
bottom_right_dst = [width - right_offset, height - bottom_offset]
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a histogram to identify the x-coordinates where the number of "lane-marking pixels" made a peak. Starting from these points, I applied a sliding window search to identify (at most) 9 points for each line. Then I fitted a 2nd degree polynomial (x = f(y)) for each line to estimate the shape of the curve. 

My pipeline is implemented in classes ColorFilter, Lane, and FrameProcessor. Identification of lane line pixels are implemented in "update" method of Lane class. The histogram search is performed only initially. For the consecutive frames, the search is started by using the line coordinates found from the previous frame. (update_using_previous method in class Lane)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The polynomials I estimated by using the Lane and FrameProcessor classes are in pixel coordinates. In order to transform to meters, I used two variables in FrameProcessor:

```python
self.ms_per_px = 3.7/700.0 #will be updated and averaged each frame
self.self.ms_per_py = 3.0/60.0 #dashed line in warped image is around 80px long
```
The variable "ms_per_px" is the resolution (meters per pixel) in x coordinates. Looking at my test results, I observed that difference between x-coordinates of right and left lines are around 700pixels, and from the specification given by course material it is around 3.7m. So this value is initialized with 3.7/700, however as the frames are processed, it is updated by the estimated distance (in pixels).

I observed that in my warped image, dashed lane markings are around 60 pixels long, so I set the y resolution to 3/60 (3m is the specified value for dashed lane marking length)

After transforming the polynomials from pixel coordinates to coordinates in meters, I calculated the radius of curvature for each line by using the formula given in https://www.intmath.com/applications-differentiation/8-radius-curvature.php

I calculated the radius of curvature by averaging the radius found from two lines.

For calculating the vehicle position, we assume the center pixel in x coordinate is the center of the vehicle. I calculated the center point of the road by calculating the intersection of the polynomials with the Y=720, which is the bottom of the warped image. Center of the lane is at (cr + cl) / 2, where cr and cl are intersection points of right and left lines, respectively.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I tested my pipeline on test images before creating the output video, to see whether if my implementation succeeded in highlighting the lane area. Below are the images for test5.jpg

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video result [https://youtu.be/jxyBxU0_iFE]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the most time consuming parts of the project was tuning the related parameters (e.g. for perspective transform, and for color thresholds) 

The pipeline did a good job to identify the lane markings and current lane of the vehicle. However the video had mostly ideal lighting conditions. It would fail when there are a lot of changing lighting conditions during the drive. One other failure case would be when nearby cars cross the lane markings. Failure to filter them out would cause erroneous detection of line parameters.


