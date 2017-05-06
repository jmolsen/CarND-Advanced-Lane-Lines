## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./output_images/distorted.png "Distorted"
[image2]: ./output_images/undistorted.png "Undistorted"
[image3]: ./output_images/undistorted_test1.png "Road Transformed"
[image4]: ./output_images/multiple_thresholds.png "Binary Example"
[image5]: ./output_images/straight_lines1_warped_plot.png "Warp Example"
[image6]: ./output_images/fit_lines.png "Fit Visual"
[image7]: ./output_images/marked_lane.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 1 through 65 of the file called `lane_lines.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Distorted Calibration Image
![Distorted Calibration Image][image1]

Undistorted Calibration Image
![Undistorted Calibration Image][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted Test Image 1][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 82 through 266 in `lane_lines.py`).  Specifically, I used a combintion of Sobel x, Sobel y, Magnitude of the Gradient with Sobel x and y, Direction of the Gradient with Sobel x and y, and Thresholding on the S-channel of the HLS version of the image.  Here's an example of my output for this step.  (note: this is from test1.jpg)

![Binary Thresholded Test Image 1][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 297 through 322 in the file `lane_lines.py`.  The `warp()` function takes as inputs an image (`img`). Source (`src`) and destination (`dst`) points values are defined outside of the function.  I chose the hardcode the source and destination points in the following manner:

```python
# Source image points
img_size = (staight_lines_image.shape[1], staight_lines_image.shape[0])
top_left = [(img_size[0] / 2) - 55, img_size[1] / 2 + 95]
bottom_left = [((img_size[0] / 6) - 15), img_size[1]]
bottom_right = [(img_size[0] * 5 / 6) + 50, img_size[1]]
top_right = [(img_size[0] / 2 + 55), img_size[1] / 2 + 95]

# Destination image points
dst_top_left = [(img_size[0] / 4), 0]
dst_bottom_left = [(img_size[0] / 4), img_size[1]]
dst_bottom_right = [(img_size[0] * 3 / 4), img_size[1]]
dst_top_right = [(img_size[0] * 3 / 4), 0]
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 455      | 320, 0        | 
| 198.33, 720   | 320, 720      |
| 1116.66, 720  | 960, 720      |
| 695, 455      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Straight Lines 1 Prespective Transform][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In lines 377 to 640 of `lane_lines.py`, I detected the lane-line pixes and fit their positions with a polynomial using Sliding Windows with a histogram to get started and then using the left and right fits to continue finding lane lines in subsequent frames of a video.  The first function is `detect_lane_sliding_window()` which takes in a binary thresholded warped image as well as instances of a `Line()` object, one for each left and right lane line to keep track of values.  The second function is `detect_lane()` which also takes in a binary thresholded warped image and two `Line()` objects.  In detect_lane, I compare the values for the last fit to the current fit and if they have any values that deviate by more than 30 pixels, I treat that as an aberration and instead use the average value of the fit from the previous 10 images from the video instead.  This should help prevent any stark changes and smooth the lane detection.

![Lane Line Pixels][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 644 through 689 in my code in `lane_lines.py` with functions `calculate_curvature()` and `calculate_postion()`.
In the calculations I defined conversions from pixels to meters using 30 meters per 720 pixels in the y dimension and 3.7 meters per 700 pixels in the x direction.  I used the formula for radius of curvature as given in the lecture:

```python
    #Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

For the vehicle position, I determined the camera center to be the center of the image in the x direction.  I calculated the lane center to be halfway between the x values for each lane line where they intersect the bottom of the image. I then calculated the difference between the camera center and the lane center and converted from pixels to meters.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 693 through 738 in my code in `lane_lines.py` in the functions `mark_lane()` and `add_curv_pos_text()`.  Here is an example of my result on a test image:

![Marked Lane][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started by consolidating the code from the lesson and the quizzes which represented the various steps needed in the pipeline along with the test plots so I could see confirm each new piece of code was working as expected.  Things mostly worked as is for the test images with only a minor tweak to the number of sliding windows from 9 to 8 which helped test image 4 get a better result.
Even the video looked mostly good until it got to a point about 3/4 through with some shadowing.  That's when I modified `detect_lane_sliding_window()` and `detect_lane()` to use the suggested `Line()` object to keep track of previous values for the lane fit.  Doing that allowed me to average out any big jumps in the lane lines so there would be smoother transitions and aberrant x values would be drowned out.  Once those changes were in place, the video seemed to perform acceptably.

