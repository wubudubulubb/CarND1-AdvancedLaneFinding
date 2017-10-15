import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import perspective



# get current directory
curr_dir = os.path.dirname(__file__)
cal_folder = 'camera_cal'
# prepare object points
nx = 9#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

directory = os.path.join(curr_dir, cal_folder)


objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

objpoints = []
imgpoints = []

for file in os.listdir(directory):
    filename = os.path.join(curr_dir, cal_folder, os.fsdecode(file))

    # read the current image into numpy array
    img = cv2.imread(filename)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # if corners found, add them to the list
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

filename = os.path.join(curr_dir, 'test_images', 'test1.jpg')
# read the current image into numpy array
img = cv2.imread(filename)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
print('e')

unwarped_image = cv2.imread(os.path.join(curr_dir, 'test_images', 'straight_lines1.jpg'))

undistorted = cv2.undistort(unwarped_image, mtx, dist, None, mtx)

width, height = 1280, 720


bottom_left = [290, 668]
top_left = [454, 557]
top_right = [838, 557]
bottom_right = [1024, 668]

src_p = np.array([bottom_left, top_left, top_right, bottom_right], np.float32)


bottom_left_dst = [140, 720]
top_left_dst = [140, 720]
top_right_dst = [1140, 600]
bottom_right_dst = [1140, 720]

plot_points = np.int32([src_p])

cv2.polylines(undistorted, plot_points, True, (255, 0, 0), 5)

dst_p = np.array([bottom_left_dst, top_left_dst, top_right_dst, bottom_right_dst], np.float32)

changed_perspective = perspective.change_perspective(undistorted, src_p, dst_p)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(unwarped_image)
ax1.set_title('Original image')
ax2.imshow(undistorted)
ax2.set_title('Calibration correction applied')
ax3.imshow(changed_perspective)
ax3.set_title('Perspective applied')
plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.)
print('e')