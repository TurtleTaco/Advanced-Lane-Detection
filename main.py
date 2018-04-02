
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
import sys

__first_frame__ = True
__test_image__ = False
__report_intermediate_result = False
__draw_sliding_window__ = False

left_fit = np.zeros(3)
right_fit = np.zeros(3)
first_N_frames_count = 0
N = 45
left_fitx_memory = [None] * N
right_fitx_memory = [None] * N
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    ''' Converting the result to unit8 displays the image correctly '''
    color_binary = np.uint8(np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255)
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Image undistortion, gradient thresholing (call pipeline) and perspective transform
def corners_unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # gradient threshold, input rgb output binary
    color_binary, combined_binary = pipeline(undist)

    # mask the binary gradient output
    mask_region = np.array([[(100, 720),(600, 400), (700, 400), (1200, 720)]], dtype=np.int32)
    masked_gradient = region_of_interest(combined_binary, mask_region)

    img_size = (masked_gradient.shape[1], masked_gradient.shape[0])
    print(img_size)
    # define source and destination points
    # src = np.float32([[626, 430], [658, 430], [1101, 720], [232, 720]])
    # dst = np.float32([[232, 0], [1101, 0], [232, 720], [1101, 720]])
    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(masked_gradient, M, img_size)
    return undist, warped, M

def read_discortion_coefficient():
    dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

def process_image(image):
    global left_fit
    global right_fit
    global __first_frame__
    global mtx
    global dist
    global left_fitx_memory
    global right_fitx_memory
    global first_N_frames_count
    global N
    global ym_per_pix
    global xm_per_pix

    font = cv2.FONT_HERSHEY_SIMPLEX
    curvature_position = (10, 660)
    offset_position = (10, 700)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    undist, warpped, perspective_matrix = corners_unwarp(image, mtx, dist)
    # Converting color space for display
    undist = cv2.cvtColor(undist, cv2.COLOR_RGB2BGR)

    ''' Detecting Lane Lines with Sliding Windows '''
    binary_warped = warpped

    if __first_frame__:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        ''' histogram is (1280,) '''
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        ''' histogram[:midpoint] is (640,) '''
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        ''' histogram[:midpoint] is (640,), index in the local right part add midpoint is actual index '''

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        # all positions of possible lane from gradient filter
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # peak x position, current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin from peak x position
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
            if __draw_sliding_window__:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            ''' ? '''
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Curverture calculation
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        average_curvature = (left_curverad + right_curverad)/2

        ''' Visualizing '''
        # Generate x and y values for plotting

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Computing the middle point of lanes
        left_lane_buttom_x = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
        right_lane_buttom_x = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2]
        mid_lane_x = (left_lane_buttom_x + right_lane_buttom_x)/2
        vehicle_offset_x = abs(mid_lane_x - 640) # in pixels
        vehicle_offset_x_meters = vehicle_offset_x * xm_per_pix

        # Caching left_fitx and right_fitx
        left_fitx_memory[0] = left_fitx
        right_fitx_memory[0] = right_fitx
        first_N_frames_count += 1

        # highlight gradient output "maybe lane"
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if __report_intermediate_result:
            plt.figure()
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.imsave('lane_detection.jpg', out_img)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(perspective_matrix), (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Curverture
        cv2.putText(result, "Curvature: " + str(float("{0:.2f}".format(average_curvature))) + "m", curvature_position, font, fontScale, fontColor, lineType)
        cv2.putText(result, "Offset: " + str(float("{0:.2f}".format(vehicle_offset_x_meters))) + "m", offset_position, font, fontScale, fontColor, lineType)

        # plt.imshow(result)
        # plt.show()
        __first_frame__ = False
        return result

    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Curverture calculation
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        average_curvature = (left_curverad + right_curverad)/2

        # Generate x and y values for plotting

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Computing the middle point of lanes
        left_lane_buttom_x = left_fit[0] * 719 ** 2 + left_fit[1] * 719 + left_fit[2]
        right_lane_buttom_x = right_fit[0] * 719 ** 2 + right_fit[1] * 719 + right_fit[2]
        mid_lane_x = (left_lane_buttom_x + right_lane_buttom_x) / 2
        vehicle_offset_x = abs(mid_lane_x - 640)  # in pixels
        vehicle_offset_x_meters = vehicle_offset_x * xm_per_pix

        # Caching left_fitx and right_fitx
        if first_N_frames_count < N:
            # print("Stacking")
            left_fitx_memory[first_N_frames_count] = left_fitx
            right_fitx_memory[first_N_frames_count] = right_fitx
            first_N_frames_count += 1
        else:
            # print("Stabilizing")
            left_fitx_memory.pop(0)
            right_fitx_memory.pop(0)
            left_fitx_memory.append(left_fitx)
            right_fitx_memory.append(right_fitx)
            # Retrieve the average coordinates from the previous N frames
            sum_left_fitx = np.zeros(720)
            sum_right_fitx = np.zeros(720)

            for left in left_fitx_memory:
                sum_left_fitx += left
            for right in right_fitx_memory:
                sum_right_fitx += right

            # print(sum_left_fitx)
            left_fitx = sum_left_fitx/N
            right_fitx = sum_right_fitx/N


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(perspective_matrix), (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        cv2.putText(result, "Curvature: " + str(float("{0:.2f}".format(average_curvature))) + "m", curvature_position, font, fontScale, fontColor, lineType)
        cv2.putText(result, "Offset: " + str(float("{0:.2f}".format(vehicle_offset_x_meters))) + "m", offset_position, font, fontScale, fontColor, lineType)
        # plt.imshow(result)
        # plt.show()
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


if __test_image__:
    mtx, dist = read_discortion_coefficient()
    img = cv2.imread('test_images/hard.jpg')
    processed_image = process_image(img)
    plt.imshow(processed_image)
    plt.show()
    plt.imsave('final.jpg', processed_image)

else:
    mtx, dist = read_discortion_coefficient()
    white_output = 'test_videos_output/project_video.mp4'
    # clip1 = VideoFileClip("project_video.mp4").subclip(0, 10)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)