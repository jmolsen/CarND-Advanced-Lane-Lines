###
### CALIBRATE CAMERA
###
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Read in and make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
c_w = 9 # number of interior corners wide
c_h = 6 # number of interior corners high

# Arrays to store object points and images points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (2,0,0), ..., (7,5,0)
objp = np.zeros((c_w*c_h,3), np.float32)
objp[:,:2] = np.mgrid[0:c_w,0:c_h].T.reshape(-1,2) # x, y coordinates

for fname in images:
    # Read in each image
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (c_w,c_h), None)
    
    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (c_w,c_h), corners, ret)
        plt.figure()
        plt.imshow(img) 
   
# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistort an image
distorted = mpimg.imread('./camera_cal/calibration1.jpg')
mpimg.imsave('./output_images/distorted.png', distorted, format='png')
undistorted = cv2.undistort(distorted, mtx, dist, None, mtx)
mpimg.imsave('./output_images/undistorted.png', undistorted, format='png')

# Draw and display the distorted and undistorted image
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(distorted)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
plt.close()

def undistort(distorted_image):
    return cv2.undistort(distorted_image, mtx, dist, None, mtx)



###
### APPLY A DISTORTION CORRECTION TO A TEST IMAGE
###
distorted_test_image = mpimg.imread('./test_images/test1.jpg')
test_image = undistort(distorted_test_image)
mpimg.imsave('./output_images/undistorted_test1.png', test_image, format='png')
plt.figure()
plt.title('Undistorted Test Image')
plt.imshow(test_image)
plt.show()
plt.close()


###
### CREATE A THRESHOLDED BINARY IMAGE USING COLOR TRANSFORMS AND GRADIENTS
###

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    x = 1 if orient=='x' else 0
    y = 1 if orient=='y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
# Run the function
gradx = abs_sobel_thresh(test_image, orient='x', thresh_min=20, thresh_max=100)
grady = abs_sobel_thresh(test_image, orient='y', thresh_min=20, thresh_max=100)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Thresholded Gradient X', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grady, cmap='gray')
ax2.set_title('Thresholded Gradient Y', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelxy = ((sobelx)**2 + (sobely)**2)**(0.5)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
# Run the function
mag_binary = mag_thresh(test_image, sobel_kernel=9, mag_thresh=(30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
# Run the function
dir_binary = dir_threshold(test_image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
    
hls_binary = hls_select(test_image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Define a function that thresholds the S-channel of HSV
# Use exclusive lower bound (>) and inclusive upper (<=)
def hsv_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 2) Apply a threshold to the S channel
    S = hsv[:,:,1]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
    
hsv_binary = hsv_select(test_image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hsv_binary, cmap='gray')
ax2.set_title('Thresholded S2', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Combine multiple thresholds
def combined_thresholds(image):
    gradx = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_binary = hls_select(image, thresh=(90, 255))
    combined = np.zeros_like(hls_binary)
    combined[(((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))) | (hls_binary == 1)] = 1
    return combined

combined = combined_thresholds(test_image)
        
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Multiple Thresholds', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
plt.close()



###
### APPLY A PERSPECTIVE TRANSFORM TO RECTIFY BINARY IMAGE ("BIRDS-EYE VIEW") 
###
distorted_staight_lines_image = mpimg.imread('./test_images/straight_lines1.jpg')
staight_lines_image = undistort(distorted_staight_lines_image)

# Source image points
img_size = (staight_lines_image.shape[1], staight_lines_image.shape[0])
top_left = [(img_size[0] / 2) - 55, img_size[1] / 2 + 95]
bottom_left = [((img_size[0] / 6) - 15), img_size[1]]
bottom_right = [(img_size[0] * 5 / 6) + 50, img_size[1]]
top_right = [(img_size[0] / 2 + 55), img_size[1] / 2 + 95]
print('top_left=', top_left)
print('bottom_left=', bottom_left)
print('bottom_right=', bottom_right)
print('top_right=', top_right)

# Destination image points
dst_top_left = [(img_size[0] / 4), 0]
dst_bottom_left = [(img_size[0] / 4), img_size[1]]
dst_bottom_right = [(img_size[0] * 3 / 4), img_size[1]]
dst_top_right = [(img_size[0] * 3 / 4), 0]
print('dst_top_left=', dst_top_left)
print('dst_bottom_left=', dst_bottom_left)
print('dst_bottom_right=', dst_bottom_right)
print('dst_top_right=', dst_top_right)

# Define perspective transform function
def warp(img):
    
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])
    
    # Four source coordinates
    src = np.float32(
            [top_left,
             bottom_left,
             bottom_right,
             top_right])
    # Four destination coordinates
    dst = np.float32(
            [dst_top_left,
             dst_bottom_left,
             dst_bottom_right,
             dst_top_right])
    
    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

# Define inverse perspective transform function
def unwarp(img):
    
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])
    
    # Four source coordinates
    src = np.float32(
            [top_left,
             bottom_left,
             bottom_right,
             top_right])
    # Four destination coordinates
    dst = np.float32(
            [dst_top_left,
             dst_bottom_left,
             dst_bottom_right,
             dst_top_right])
    
    # Could compute the inverse by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Create warped image - uses linear interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    return unwarped

# Get perspective transform
warped_im = warp(staight_lines_image)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(staight_lines_image)
ax1.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], [top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], color='r', marker = '.')
ax1.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], [bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], color='r', marker = '.')
ax2.set_title('Warped image')
ax2.imshow(warped_im)
ax2.plot([dst_top_left[0], dst_bottom_left[0]], [dst_top_left[1], dst_bottom_left[1]], [dst_top_right[0], dst_bottom_right[0]], [dst_top_right[1], dst_bottom_right[1]], color='r', marker = '.')
ax2.plot([dst_top_left[0], dst_top_right[0]], [dst_top_left[1], dst_top_right[1]], [dst_bottom_left[0], dst_bottom_right[0]], [dst_bottom_left[1], dst_bottom_right[1]], color='r', marker = '.')
plt.show()

unwarped_im = unwarp(warped_im)
plt.title('Unwarped image')
plt.imshow(unwarped_im)
plt.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], [top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], color='r', marker = '.')
plt.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], [bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], color='r', marker = '.')
plt.show()
plt.close()



###
### DETECT LANE PIXELS AND FIT TO FIND THE LANE BOUNDARY
###

binary_warped = warp(combined)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = []  
        #y values for detected line pixels
        self.ally = [] 
        
left_line = Line()
right_line = Line()

# Detect lane pixels and fit to find the lane boundary using sliding windows
def detect_lane_sliding_window(binary_warped, left_line, right_line):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if (len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if (len(good_right_inds) > minpix):       
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    left_line.allx.extend(leftx)
    lefty = nonzeroy[left_lane_inds] 
    left_line.ally.extend(lefty)
    rightx = nonzerox[right_lane_inds]
    right_line.allx.extend(rightx)
    righty = nonzeroy[right_lane_inds]
    right_line.ally.extend(righty)
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_line.current_fit.insert(0, left_fit)
    right_line.current_fit.insert(0, right_fit)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_line.recent_xfitted.insert(0, left_fitx)
    right_line.recent_xfitted.insert(0, right_fitx)
        
    left_line.detected = True
    right_line.detected = True
    
    return left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit


left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit = detect_lane_sliding_window(binary_warped, left_line, right_line)
    
### Sliding Window Visualization
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_line.recent_xfitted[0]
right_fitx = right_line.recent_xfitted[0]

out_img[lefty[0], leftx[0]] = [255, 0, 0]
out_img[righty[0], lefty[0]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()

    
# Detect lane pixels and fit to find the lane boundary using values found using sliding windows
def detect_lane(binary_warped, left_line, right_line):    
    left_fit = left_line.current_fit[0]
    right_fit = right_line.current_fit[0]
    ### Skip the sliding windows step once you know where the lines are
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 

    left_line.allx.extend(leftx)
    left_line.ally.extend(lefty)
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    right_line.allx.extend(rightx)
    right_line.ally.extend(righty)
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    max_best_count = 10
    
    best_count = 0
    left_line.best_fit = np.zeros(3)
    for left_cf in left_line.current_fit[:max_best_count]:
        left_line.best_fit += left_cf
        best_count += 1
    left_line.best_fit = left_line.best_fit / best_count
    
    best_count = 0
    right_line.best_fit = np.zeros(3)
    for right_cf in right_line.current_fit[:max_best_count]:
        right_line.best_fit += right_cf
        best_count += 1
    right_line.best_fit = right_line.best_fit / best_count 
    
    left_line.current_fit.insert(0, left_fit)
    right_line.current_fit.insert(0, right_fit)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_line.diffs = left_line.recent_xfitted[0] - left_fitx
    right_line.diffs = right_line.recent_xfitted[0] - right_fitx
    
    best_count = 0
    left_line.bestx = np.zeros(720)
    for left_rxf in left_line.recent_xfitted[:max_best_count]:
        left_line.bestx += left_rxf
        best_count += 1
    left_line.bestx = left_line.bestx / best_count
    
    best_count = 0
    right_line.bestx = np.zeros(720)
    for right_rxf in right_line.recent_xfitted[:max_best_count]:
        right_line.bestx += right_rxf
        best_count += 1
    right_line.bestx = right_line.bestx / best_count   
    
    max_x_diff = 30
    if any(l_x_val > max_x_diff for l_x_val in left_line.diffs):
        left_fit = left_line.best_fit
        left_fitx = left_line.bestx
    if any(r_x_val > max_x_diff for r_x_val in right_line.diffs):
        right_fit = right_line.best_fit
        right_fitx = right_line.bestx
         
    left_line.current_fit.insert(0, left_fit)
    right_line.current_fit.insert(0, right_fit)
    left_line.recent_xfitted.insert(0, left_fitx)
    right_line.recent_xfitted.insert(0, right_fitx)
    
    return left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty


left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = detect_lane(binary_warped, left_line, right_line)
    
### Final Visualization
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[lefty[0], leftx[0]] = [255, 0, 0]
out_img[righty[0], lefty[0]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
margin = 100
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
plt.close()



###
### DETERMINE THE CURVATURE OF THE LANE AND VEHICLE POSITION WITH RESPECT TO CENTER
###

# Plot up the data
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
plt.show()

def calculate_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

left_curverad, right_curverad = calculate_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)
print(left_curverad, 'm', right_curverad, 'm')

def calculate_position(leftx, rightx):
    # Calculate vehicle position with respect to center
    lane_center = leftx[-1] + ((rightx[0]-leftx[-1])/2)
    camera_center = 1280/2
    
    # Define conversion in x from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    vehicle_position = (camera_center - lane_center) * xm_per_pix
    return vehicle_position
                       
vehicle_position = calculate_position(leftx, rightx)             
print(vehicle_position, 'm')


        
###
### WARP THE DETECTED LANE BOUNDARIES BACK ONTO THE ORIGINAL IMAGE
###       

def mark_lane(warped_image, orig_image, left_fitx, right_fitx, ploty): 
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv) 
    newwarp = unwarp(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    return result

marked_lane = mark_lane(binary_warped, test_image, left_fitx, right_fitx, ploty)

def add_curv_pos_text(marked_lane_image, curverad, vehicle_position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    radius_of_curvature = "Radius of Curvature = %d (m)" % (curverad)
    result_with_curv = cv2.putText(marked_lane_image, radius_of_curvature, (50, 50), font, font_scale, color, thickness)
    vehicle_side = 'right'
    if (vehicle_position < 0):
        vehicle_side = 'left'
        vehicle_position = -1*vehicle_position
    distance_from_center = "Vehicle is %.2f (m) %s of center" % (vehicle_position, vehicle_side)
    if (vehicle_position == 0):
        distance_from_center = "Vehicle is at the center of the lane"
    result_with_curv_pos = cv2.putText(result_with_curv, distance_from_center, (50, 100), font, font_scale, color, thickness)
    return result_with_curv_pos

final_image = add_curv_pos_text(marked_lane, left_curverad, vehicle_position)
plt.imshow(final_image)
plt.show()
plt.close()


        
###       
### Lane Marking Pipeline
###
      
def lane_marking_pipeline(image, left_line, right_line):
    ### APPLY A DISTORTION CORRECTION
    undistorted_image = undistort(image)
    ### CREATE A THRESHOLDED BINARY IMAGE USING COLOR TRANSFORMS AND GRADIENTS
    binary_thresholded = combined_thresholds(undistorted_image)
    ### APPLY A PERSPECTIVE TRANSFORM TO RECTIFY BINARY IMAGE ("BIRDS-EYE VIEW") 
    binary_warped = warp(binary_thresholded)
    ### DETECT LANE PIXELS AND FIT TO FIND THE LANE BOUNDARY
    if not left_line.detected or not right_line.detected:
        left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit = detect_lane_sliding_window(binary_warped, left_line, right_line)
    left_line, right_line, left_lane_inds, right_lane_inds, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = detect_lane(binary_warped, left_line, right_line)
    ### DETERMINE THE CURVATURE OF THE LANE AND VEHICLE POSITION WITH RESPECT TO CENTER
    left_curverad, right_curverad = calculate_curvature(left_line.current_fit[0], right_line.current_fit[0], left_line.recent_xfitted[0], right_line.recent_xfitted[0], ploty)
    vehicle_position = calculate_position(leftx, rightx)
    ### WARP THE DETECTED LANE BOUNDARIES BACK ONTO THE ORIGINAL IMAGE
    marked_lane = mark_lane(binary_warped, undistorted_image, left_line.recent_xfitted[0], right_line.recent_xfitted[0], ploty)
    final_image = add_curv_pos_text(marked_lane, left_curverad, vehicle_position)
    return final_image

# Test lane marking pipeline on test images
test_images = glob.glob('./test_images/*.jpg')
for fname in test_images:
    # Read in each image
    img = mpimg.imread(fname)  
    left_line = Line()  
    right_line = Line()
    marked_img = lane_marking_pipeline(img, left_line, right_line)
    filename = "./output_images/%s_lane_lines.png" % (fname.split('/')[-1].split('.')[0])
    mpimg.imsave(filename, marked_img, format='png')
    plt.figure()
    plt.imshow(marked_img)
    plt.show()
    
    
# Run lane markin pipeline on videos
from moviepy.editor import VideoFileClip

def lane_marking_clip_pipeline(clip, left_line, right_line):
    def lane_marking_clip_param_pass(image):
        return lane_marking_pipeline(image, left_line, right_line)
    return clip.fl_image(lane_marking_clip_param_pass)

left_line = Line() 
right_line = Line()
project_video_output = './project_video_output.mp4'
project_video_clip = VideoFileClip("./project_video.mp4")  
project_video_output_clip = project_video_clip.fx(lane_marking_clip_pipeline, left_line, right_line)
project_video_output_clip.write_videofile(project_video_output, audio=False)

#left_line = Line() 
#right_line = Line()
#project_video_output = './challenge_video_output.mp4'
#project_video_clip = VideoFileClip("./challenge_video.mp4")  
#project_video_output_clip = project_video_clip.fx(lane_marking_clip_pipeline, left_line, right_line)
#project_video_output_clip.write_videofile(project_video_output, audio=False)

#left_line = Line() 
#right_line = Line()
#project_video_output = './harder_challenge_video_output.mp4'
#project_video_clip = VideoFileClip("./harder_challenge_video.mp4")  
#project_video_output_clip = project_video_clip.fx(lane_marking_clip_pipeline, left_line, right_line)
#project_video_output_clip.write_videofile(project_video_output, audio=False)



   
