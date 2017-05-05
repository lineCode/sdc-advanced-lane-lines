import cv2
import numpy as np
import os
import glob
import pickle

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


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
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.points = None


class LaneLines(object):
    # TODO: Make this a singleton
    def __init__(self):
         # Set directories
        self.cal_dir = "camera_cal"
        self.results_dir = "results"

        # Initialize New Lanes
        self.left = Line()
        self.right = Line()

        # Set Calibration defaults
        self.dist_mtx = None
        self.dist_dist = None
        self.cal_file = os.path.join(self.results_dir, "calibration_data.p")

        # Set the perspective transform points
        # They are broken down into src/dst top/bottom/left/right variables
        # To provide clarity
        # XXX: These were tuned on the test_image and assumer all images 
        # XXX: are of the same viewing angle and dimensions (front-facing, 1280x720)
        x = 1280
        y = 717
        off = 50

        br = [1044, 669]
        bl = [235, 675]
        tr = [730, 455]
        tl = [555, 460]

        dbr = [x - off, y - off]
        dbl = [0 + off, y - off]
        dtr = [x - off, 0 + off]
        dtl = [0 + off, 0 + off]

        self.trans_src = np.float32([bl, tl, tr, br])
        self.trans_dst = np.float32([dbl, dtl, dtr, dbr])
        self.trans_M = None
        self.trans_M_rev = None

        # Calibrate the camera and setup transformqtion matrix
        self.calibrate()
        self.setup_transform()

    def calibrate(self, x = 9, y = 6, debug = False, read_cal = True):
        '''Wrapper for calibrate_camera'''
        # Run calibration with all images in camera_cal/calibration*.jpg)
        # Read data from calibration_data.p if already calculated, else write it
        calibration_images = glob.glob(os.path.join(self.cal_dir, 
                "calibration*.jpg"))
        self.calibrate_camera(calibration_images, x, y, debug, read_cal)

    def calibrate_camera(self, images, x, y, debug, read_cal):
        '''Take a list of chessboard calibration images and calibrate params
        Input must be RGB image files of x by y chessboards.
        If read_cal is specified, will try to pickle load data instead of calculate.
        '''
        # Try to read calibration data in first
        if read_cal:
            print("Reading pre-calculated calibration data")
            try:
                cal_data = pickle.load(open(self.cal_file, "rb" ))
                self.dist_mtx = cal_data['dist_mtx']
                self.dist_dist = cal_data['dist_dist']
                return
            except (IOError, KeyError) as e:
                print("Unable to read calibration data from %s ... \
                 preceeding to calculate" %(read_cal))

        # Setup variables and do basic checks
        assert images is not None and len(images) > 0
        objpoints = []
        imgpoints = []
        img_shape = None

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((y*x,3), np.float32)
        objp[:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1,2)

        # Iteratere over each image and try to calibrate points on checkerboards
        for idx, fname in enumerate(images):
            print("Calibrating against image %d:%s" %(idx, fname))
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_shape is None:
                img_shape = gray.shape

            # TODO: What should these look like? assert gray.shape == img_shape
            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
            
            # TODO: Do we want to retry a different size?
            # IF we found corners, update the lists and do some debug
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                if debug:
                    cv2.drawChessboardCorners(img, (x,y), corners, ret)
                    write_name = os.path.join(self.results_dir, 
                            'corners_found' + str(idx) + '.jpg')
                    cv2.imwrite(write_name, img)
            else:
                print("No corners found in image.")
        print("Finished calibration images ... Running calibration algorithm.")

        # Calibrate distortion matrix based on imaage points
        ret, self.dist_mtx, self.dist_dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                img_shape, None, None)

        # Save undistorted images
        if debug:
            for idx, fname in enumerate(images):
                img = cv2.imread(fname)
                img = self.correct_distortion(img)
                #img = self.perspective_transform(img)
                write_name = os.path.join(self.results_dir, 
                            'undistort' + str(idx) + '.jpg')
                cv2.imwrite(write_name, img)

        # Save data
        cal_data = {'dist_mtx': self.dist_mtx, 'dist_dist': self.dist_dist}
        dist_pickle = pickle.dump(cal_data, open(self.cal_file, "wb" ))

    def setup_transform(self):
        '''Use the pre-calculated points to save the transformation matrix'''
        self.trans_M = cv2.getPerspectiveTransform(self.trans_src, self.trans_dst)
        self.trans_M_rev = cv2.getPerspectiveTransform(self.trans_dst, self.trans_src)

    def process_video(self, input_vid, output_vid):
        '''Run an video through the pipeline'''
        print("Running %s through pipeline and outputting to %s" %(input_vid, output_vid))
        clip = VideoFileClip(input_vid)
        output = clip.fl_image(self.pipeline)
        output.write_videofile(output_vid, audio=False)

    def pipeline(self, img):
        '''run an image through the full pipeline and return a lane-filled image'''
        img = self.correct_distortion(img)
        mask = self.edge_detection(img)
        img[mask != 1] = 0
        img = self.perspective_transform(img)

        img = self.find_lanes_conv(img)
        plt.imshow(img); plt.show()

        img = self.perspective_transform(img, rev = True)
        plt.imshow(img); plt.show()
        return img
        raise NotImplemented

    def perspective_transform(self, img, rev=False):
        '''Transform the perspective'''
        # XXX: For some reason img shape coordinates need to be flipped here
        if rev:
            return cv2.warpPerspective(img, self.trans_M_rev, (img.shape[1], img.shape[0]), 
                    flags=cv2.INTER_LINEAR)  
        return cv2.warpPerspective(img, self.trans_M, img.shape[-2::-1], 
                flags=cv2.INTER_LINEAR)    

    def correct_distortion(self, img):
        '''Given an image, use pre-calculated correction/distortion matrices to undistort the image'''
        assert self.dist_mtx is not None
        assert self.dist_dist is not None
        assert img is not None
        undistort = cv2.undistort(img, self.dist_mtx, self.dist_dist)
        assert img.shape == undistort.shape
        return undistort

    def edge_detection(self, img):
        mask = np.zeros(img.shape[:-1])

        # Convert to HLS and take interest S values
        color_mask = self.color_thresh(img, thresh=(90, 255))
        img[color_mask != 1 ] = 0

        # Remove anything to slanted
        y_mask = self.gradient_thresh(img, orient = 'y', thresh=(30,200))

        # Remove anything to left/right leaning
        x_mask = self.gradient_thresh(img, thresh=(0,200), orient = 'x')

        # Cut off top and sides of image
        location_mask = self.location_thresh(img, thresh=(0.6, 0.4))

        #dir_mask = self.dir_thresh(img, thresh=(0.5, 2.3))
        # mag_mask = self.magnitude_thresh(img, thresh=(0, 100))

        mask[(x_mask == 1) & (color_mask == 1) & 
             (y_mask == 1) & (location_mask == 1)] = 1
        return mask 

    def location_thresh(self, img, thresh=(0.3, 0.3)):
        '''Cut of the thresh %% top of the image'''
        mask = np.zeros(img.shape[:-1])

        #bottom = img.shape[0]
        #y = int(bottom * thresh) # y cut-off
        #mask[y:bottom,:] = 1

        #filling pixels inside the polygon defined by "vertices" with the fill color 
        poly_x = img.shape[1]
        poly_y = img.shape[0]  

        vertices = np.array([[(0 + poly_x * thresh[1],poly_y * thresh[0]),
                (poly_x - poly_x * thresh[1], poly_y * thresh[0]), # A little bit left of center
                (poly_x, poly_y),
                (0, poly_y)]], dtype=np.int32)
        ignore_mask_color = 1 
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        return mask

    def color_thresh(self, img, thresh=(50, 255)):
        '''Return an img based on color threshold
        Converts img to HLS and then uses Saturation levels for threshold
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]

        mask = np.zeros_like(S)
        mask[(S > thresh[0]) & (S <= thresh[1])] = 1

        return mask

    def magnitude_thresh(self, img, thresh=(50, 120), sobel_kernel=3):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Generate sobel
        x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        # Calculate scaled magnitude
        magnitude = np.sqrt(x_sobel ** 2 + y_sobel ** 2)
        scaled_magnitude = 255 * magnitude / np.max(magnitude)
        
        # Mask based on threshold
        mask = np.zeros(gray.shape)
        mask[(scaled_magnitude > thresh[0]) & (scaled_magnitude < thresh[1])] = 1
        
        return mask

    def gradient_thresh(self, img, orient='x', thresh=(0, 15), sobel_kernel=15):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Take derif in respect to orient
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        if orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
            
        # Take absolute value
        sobel = np.absolute(sobel)
        
        # Scale to 0-255 and cast to int
        sobel = 255 * sobel / np.max(sobel)
        sobel = np.uint8(sobel)
        
        # Create a mask with 1s where the threshold is met
        mask = np.zeros(gray.shape)
        mask[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
        
        return mask

    def dir_thresh(self, img, thresh=(0, np.pi/2), sobel_kernel=3):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Take sobel gradients
        x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_x = np.absolute(x_sobel)
        abs_y = np.absolute(y_sobel)
        
        # Calculate direction
        direction = np.arctan2(abs_y, abs_x)
        
        # Create a threshold based mask
        mask = np.zeros(gray.shape)
        mask[(direction > thresh[0]) & (direction < thresh[1])] = 1

        return mask

    def find_lanes_conv(self, img):
        '''Take a edge detected, perspective transformed image and detect lines
        Will update self.left and self.right with correct pixel lists, fit lines, etc.
        '''
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[img != 0] = 1

        w_width = 50
        w_height = 80
        margin = 50

        levels = int(img.shape[0] / w_height)
        offset = w_width / 2

        w_centroids = []
        w = np.ones(w_width)

        img_bottom = int(img.shape[0] / 4 * 3)
        img_center = int(img.shape[1] / 2)

        # Bottom left corner
        l_sum = np.sum(img[img_bottom:, :img_center], axis=0)
        l_center = np.argmax(np.convolve(w, l_sum)) - w_width / 2

        # Bottom right corner
        r_sum = np.sum(img[img_bottom:, img_center:], axis=0)
        r_center = np.argmax(np.convolve(w,r_sum)) - w_width / 2 + img_center
    
        w_centroids.append(((l_center, w_height * 0.5), (r_center, w_height * 0.5)))

        for level in range(1, levels):
            y_max = (level + 1) * w_height
            y_min = level * w_height
            y_center = (level + 0.5) * w_height

            img_layer = np.sum(img[int(img.shape[0] - y_max):int(img.shape[0] - y_min), :], axis=0)
            conv_signal = np.convolve(w, img_layer)

            l_min_idx = int(max(l_center + offset - margin, 0))
            l_max_idx = int(min(l_center + offset + margin, img.shape[1]))
            l_center =  np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset

            r_min_idx = int(max(r_center + offset - margin, 0))
            r_max_idx = int(min(r_center + offset + margin, img.shape[1]))
            r_center =  np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset

            w_centroids.append(((l_center, y_center), (r_center, y_center)))

        if len(w_centroids) <= 0:
            return img



        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        for level in range(0, len(w_centroids)):
            l_mask = window_mask(w_width, w_height, img, w_centroids[level][0][0], level)
            l_points[l_mask == 1] = 1

            r_mask = window_mask(w_width, w_height, img, w_centroids[level][1][0], level)
            r_points[r_mask == 1] = 1

        img = np.zeros_like(img)

        img[(l_points == 1) | (r_points == 1)] = 1
        
        self.left.points = [c[0] for c in w_centroids]
        self.right.points = [c[1] for c in w_centroids]

        # Fit a second order polynomial to each
        left_x = [p[1] for p in self.left.points]
        left_y = [p[0] for p in self.left.points]
        self.left.best_fit = np.polyfit(left_x, left_y, 2)

        right_x = [p[1] for p in self.right.points]
        right_y = [p[0] for p in self.right.points]
        self.right.best_fit = np.polyfit(right_x, right_y[::-1], 2)
        print("XXXXX")
        print(right_y)



        plot = np.linspace(0, img.shape[0] - 1, img.shape[0] )
        plot = plot[::-1]
        self.left.best_fit  = (self.left.best_fit [0]**2) * plot + self.left.best_fit [1]*plot + self.left.best_fit [2]
        self.right.best_fit = (self.right.best_fit[0]**2) * plot + self.right.best_fit[1]*plot + self.right.best_fit[2]

        self.left.best_fit[self.left.best_fit <  0] = 0

        self.right.best_fit[self.right.best_fit > img.shape[1]] = img.shape[1]
        #print(self.right.best_fit)


        #plt.plot(left_y, left_x, color='red')
        #plt.plot(right_y[::-1], right_x, color='green')

        plt.plot(self.left.best_fit, plot, color='red')
        plt.plot(self.right.best_fit, plot, color='green')

        #img = cv2.fillPoly(img, self.left.best_fit + self.right.best_fit, (0,255, 0))
        #plt.imshow(img*255)
        #plt.show()

        return img


    def find_lanes_junk(self, binary_warped, debug=False):
        '''This was mostly bogus sample code and should not be used or looked at'''
        if len(self.right.recent_xfitted) == 0 or len(self.left.recent_xfitted == 0):
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped, axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
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

            # Generate x and y values for plotting
            if debug:
                ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
                plt.imshow(out_img)
                plt.plot(left_fitx, ploty, color='yellow')
                plt.plot(right_fitx, ploty, color='yellow')
                plt.xlim(0, 1280)
                plt.ylim(720, 0)
                plt.show()

            self.left.best_fit = left_fit
            self.right.best_fit = right_fit

            self.left.current_fit = left_fit
            self.right.current_fit = right_fit

            self.left.recent_fit = left_fit
            self.right.current_fit = right_fit

            self.left.detected = len(left_lane_inds) > 0
            self.right.detected = len(right_lane_inds) > 0

            self.left.bestx = leftx
            self.right.bestx = rightx
            
            self.left.allx = leftx
            self.right.allx = rightx

            self.left.ally = lefty
            self.right.ally = righty

            self.diffs = np.array([0,0,0], dtype='float') 
            '''
            # x values of the last n fits of the line
            self.recent_xfitted = [] 
      
     
            #radius of curvature of the line in some units
            self.radius_of_curvature = None 
            #distance in meters of vehicle center from the line
            self.line_base_pos = midpoint 

            '''

        else: # TODO:
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
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            if debug:
                # Create an image to draw on and an image to show the selection window
                out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
                window_img = np.zeros_like(out_img)
                # Color in left and right line pixels
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
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
                plt.xlim(0, img.shape[0])
                plt.ylim(img.shape[1], 0)


        # TODO: def lane_curvature(self, img, debug):
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        if debug:
            mark_size = 3
            plt.plot(leftx, lefty, 'o', color='red', markersize=mark_size)
            plt.plot(rightx, righty, 'o', color='blue', markersize=mark_size)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(left_fitx, ploty, color='green', linewidth=3)
            plt.plot(right_fitx, ploty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.show()

        self.detected = False  
        self.recent_xfitted = [] 
        self.bestx = None     
        self.best_fit = None  
        self.current_fit = [np.array([False])]  
        self.radius_of_curvature = None 
        self.line_base_pos = None 
        self.diffs = np.array([0,0,0], dtype='float') 
        self.allx = None  
        self.ally = None

    def calculate_curvature(self, y, A, B):
        # TODO: convert px values
        return (1 + ( (2 * A * y + B) ** 2) ** (3 / 2) ) / np.absolute(2 * A)

    def fill_lanes(self, img):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

if __name__ == '__main__':

    lane_lines = LaneLines()
    ''''
    img = cv2.imread(os.path.join("test_img", "test1.jpg"))
    img_blank = np.zeros_like(img)
    # Test Image Transformation
    transform_img = cv2.imread(os.path.join("test_img", "transform_test.jpg"))
    
    # Test calibration and update calibration data file
    #lane_lines.calibrate(debug = True, read_cal = False)

    img_blank = np.zeros_like(img)

    undistort = lane_lines.correct_distortion(img)
    write_name = os.path.join("results", "test-undistort.jpg")
    cv2.imwrite(write_name, undistort)

    # Test color thresh
    color_mask = lane_lines.color_thresh(img)
    cv2.imwrite(os.path.join("results", "color_edge.jpg"), 255*color_mask)
    
    # Test magnitude thresh
    magnitude_mask = lane_lines.magnitude_thresh(img)
    cv2.imwrite(os.path.join("results", "magnitude_edge.jpg"), 255*magnitude_mask)

    # Test x gradient
    x_mask = lane_lines.gradient_thresh(img, orient = 'x')
    cv2.imwrite(os.path.join("results", "x_edge.jpg"), 255*x_mask)

    # Test y gradient
    y_mask = lane_lines.gradient_thresh(img, orient = 'y')
    cv2.imwrite(os.path.join("results", "y_edge.jpg"), 255*y_mask)

    # Test dir thresh
    dir_mask = lane_lines.dir_thresh(img)
    cv2.imwrite(os.path.join("results", "dir_edge.jpg"), 255*dir_mask)

    # Test location thresh
    loc_mask = lane_lines.location_thresh(img)    
    cv2.imwrite(os.path.join("results", "loc_thresh.jpg"), 255*loc_mask)

    # Test edge detection pipelien
    edge_mask = lane_lines.edge_detection(undistort)
    cv2.imwrite(os.path.join("results", "edge.jpg"), 255*edge_mask)
 
    edge_img = np.copy(undistort)
    edge_img[edge_mask != 1 ] = 0

    transform = lane_lines.perspective_transform(edge_img)
    write_name = os.path.join("results", "test-transform.jpg")
    cv2.imwrite(write_name, transform)

    img1 = cv2.imread(os.path.join("test_img", "curves1.jpg"))
    img2 = cv2.imread(os.path.join("test_img", "curves2.jpg"))
    img3 = cv2.imread(os.path.join("test_img", "curves3.jpg"))
    img = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img[img == 255] = 1
    img1[img1 == 255] = 1
    img2[img2 == 255] = 1
    img3[img3 == 255] = 1
    lane_lines.find_lanes(img, True)

    '''
    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join("results", "project_video_output.mp4")
    lane_lines.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'challenge_video.mp4')
    output_vid = os.path.join("results", "challenge_video_output.mp4")
    lane_lines.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'harder_challenge_video.mp4')
    output_vid = os.path.join("results", "harder_challenge_video_output.mp4")
    lane_lines.process_video(input_vid, output_vid)
    #lane_lines.find_lanes(img2, True)
    #lane_lines.find_lanes(img3, True)
    
    #lane_lines.lane_curvature(img, True)

    # TODO: get find_lanes_conv to calculate the line of best fit
    # TODO: Get find_lanes working on the update case
    # TODO: Get lane_curvature working
    # TODO: Get fill_lanes Working
    # TODO: Get pipeline working
    # TODO: Writeup
    # Cleanup code
