import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
import os
import numpy as np
import cv2




class ColorFilter():
    def __init__(self):
        pass

    def filter(self, original_image):
        combined = np.zeros_like(original_image[:,:,0],np.uint8)
        yellow_binary = self.yellow_mask(original_image)
        white_binary = self.white_mask(original_image)
        sat_binary = self.hsv_mask(original_image)
        color_filter = cv2.bitwise_and(original_image,original_image,
                               mask = cv2.bitwise_and(
                                   cv2.bitwise_or(yellow_binary ,white_binary),
                                   sat_binary))

        combined[((color_filter[:,:,0] >0 ) |
                   (color_filter[:,:,1] >0) |
                   (color_filter[:,:,2] >0)
                         )] = 1

        return np.dstack((combined, combined, combined))*255

    def yellow_mask(self, img):
        '''function to return masking for yellow lines'''
        yellow_max = np.array([255,255,255],np.uint8)
        yellow_min = np.array([0,180,0],np.uint8)
        yellow_binary = np.zeros_like(img[:,:,0])
        cv2.inRange(img, yellow_min, yellow_max, yellow_binary)
        return yellow_binary

    def white_mask(self, img):
        '''function to return masking for white lines'''
        white_min = np.array([150,150,150],np.uint8)
        white_max = np.array([250,250,250],np.uint8)
        white_binary = np.zeros_like(img[:,:,0])
        cv2.inRange(img, white_min, white_max, white_binary)
        return white_binary

    def hsv_mask(self, img):
        '''function to return masking for hsv filter'''
        saturation_min = np.array([0,0,100],np.uint8)
        saturation_max = np.array([255,255,255],np.uint8)
        sat_binary = np.zeros_like(img[:,:,0])
        cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HLS),
                saturation_min, saturation_max, sat_binary)
        return sat_binary


class Lane():
    def __init__(self):
        self.history_A = []
        self.history_B = []
        self.history_C = []
        self.num_lines = 0
        self.MAX_NUM_HISTORY_LINES = 10
        self.lane_inds = []
        self.nwindows = 9
        self.reset = False

    def init_plot_points(self, img):
        self.ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        self.xPts = np.zeros_like(self.ploty, np.float32)

    def A(self):
        if len(self.history_A) > 0:
            return sum(self.history_A)/self.num_lines
        else:
            return 0

    def B(self):
        if len(self.history_B) > 0:
            return sum(self.history_B)/self.num_lines
        else:
            return 0

    def C(self):
        if len(self.history_C) > 0:
            return sum(self.history_C)/self.num_lines
        else:
            return 0

    def add_line(self, a, b, c):
        if self.num_lines < self.MAX_NUM_HISTORY_LINES:
            self.num_lines += 1
        else:
            del self.history_A[0]
            del self.history_B[0]
            del self.history_C[0]
        self.history_A.append(a)
        self.history_B.append(b)
        self.history_C.append(c)

    def update(self, xbase, binary_warped_img):
        window_height = np.int(binary_warped_img.shape[0]/self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = xbase
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 10
        # Create empty lists to receive left and right lane pixel indices
        self.lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped_img.shape[0] - (window+1)*window_height
            win_y_high = binary_warped_img.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin


            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            self.lane_inds.append(good_inds)

            # If found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))


        # Concatenate the arrays of indices
        self.lane_inds = np.concatenate(self.lane_inds)


        # Extract line pixel positions
        X = nonzerox[self.lane_inds]
        Y = nonzeroy[self.lane_inds]


        # Fit a second order polynomial
        if len(X) > 2:
            polyline_fit = np.polyfit(Y, X, 2)
            self.add_line(polyline_fit[0], polyline_fit[1], polyline_fit[2])

    def update_using_previous(self, xbase, binary_warped):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        self.lane_inds = []
        self.lane_inds = ((nonzerox > (self.A()*(nonzeroy**2) + self.B()*nonzeroy +
                            self.C() - margin)) & (nonzerox < (self.A()*(nonzeroy**2) +
                            self.B()*nonzeroy + self.C() + margin)))


        # Again, extract left and right line pixel positions
        X = nonzerox[self.lane_inds]
        Y = nonzeroy[self.lane_inds]

        if len(X) > 2:
            # Fit a second order polynomial to each
            polyline_fit = np.polyfit(Y, X, 2)

            # TODO: sanity check here
            self.add_line(polyline_fit[0], polyline_fit[1], polyline_fit[2])
        else:
            self.reset = True
            self.update(xbase, binary_warped)
            self.reset = False

    def get_pts(self):
        self.xPts = np.array([(y**2)*self.A() + y*self.B() + self.C()
                              for y in self.ploty])
        return np.array([np.transpose(np.vstack([self.xPts, self.ploty]))])


class FrameProcessor():
    def __init__(self):
        self.mtx = pickle.load( open( "cal_mtx.p", "rb" ) )
        self.dist = pickle.load( open( "cal_dist.p", "rb" ) )
        self.M = pickle.load(open("cam_perspective_transform.p", "rb"))
        self.Minv = pickle.load(open("cam_perspective_transform_inv.p", "rb"))
        self.color_filter = ColorFilter()

        self.left_line = Lane()
        self.right_line = Lane()
        #self.left_line.add_line(0, 0, 280)
        #self.right_line.add_line(0,0, bottom_right_dst[0] )

        self.image_size = None
        self.blank_warped = None
        self.color_warped = None
        self.warped = None

        self.first_estimation = True

    def calculate_line_params(self, unwarped_binary_image):
        self.warped = cv2.warpPerspective(unwarped_binary_image, self.M, self.image_size, flags=cv2.INTER_LINEAR)
        binary_warped = self.warped[:,:,0]

        # from lecture notes:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped)))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if self.first_estimation:
            self.left_line.update(leftx_base, out_img)
            self.right_line.update(rightx_base, out_img)
            self.first_estimation = False
        else:
            self.left_line.update_using_previous(leftx_base, out_img)
            self.right_line.update_using_previous(rightx_base, out_img)

    def process(self, img):
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        filtered = self.color_filter.filter(undistorted)
        if self.image_size is None:
            self.image_size = (img.shape[1], img.shape[0])
            self.left_line.init_plot_points(filtered)
            self.right_line.init_plot_points(filtered)

        self.calculate_line_params(filtered)
        return self.add_line_annotation(img)

    def add_line_annotation(self, img):

        if self.blank_warped is None:
            self.blank_warped = np.zeros_like(img[:, :, 0],np.uint8)
            self.color_warped = np.dstack((self.blank_warped, self.blank_warped, self.blank_warped))

        pts_left = self.left_line.get_pts()
        pts_right = np.fliplr(self.right_line.get_pts())
        print(len(self.left_line.ploty))
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(self.color_warped, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(self.color_warped, self.Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        #result = cv2.addWeighted(self.warped, 1, self.color_warped, 0.3, 0)
        return result


curr_dir = os.getcwd()
output = os.path.join(curr_dir,'deneme.mp4')

fp = FrameProcessor()
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip(os.path.join(curr_dir,"project_video.mp4"))
clip1 = VideoFileClip(os.path.join(curr_dir,"project_video.mp4")).subclip(0,5)
clip = clip1.fl_image(fp.process) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)
