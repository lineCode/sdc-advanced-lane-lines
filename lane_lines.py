import cv2
import numpy as np
import os
import glob
import pickle

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, side, centroids):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     

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

        # Side
        self.side = side

        # Number of Centroids
        self.centroid_count = centroids

        # All of the Centroid Points
        self.fit_pts = None

        # Polynomial coefficients averaged over last detected Centroid pts
        self.best_fit = None

    def update_best_fit(self):
        '''Update the best fit polynomials based on current points'''
        x = [p[1] for p in self.points]
        y = [p[0] for p in self.points]
        # XXX: Note something funky is going on with the dimensions and I needed to flip the right lane y values
        if self.side == 'right':
            y = y[::-1]
        self.best_fit = np.polyfit(x, y, 2)



class LaneLines(object):
    # TODO: Make this a singleton
    def __init__(self):
         # Set directories
        self.cal_dir = "camera_cal"
        self.results_dir = "results"

        # Initialize New Lanes
        self.left = Line('left', 9)
        self.right = Line('right', 9)

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

    def pipeline(self, img, debug=False):
        '''run an image through the full pipeline and return a lane-filled image'''
        # undistort and create a copy
        img = self.correct_distortion(img)
        undistort_img = np.copy(img)

        # create edge detection mask, and zero out anything not found in mask
        mask = self.edge_detection(img)
        img[mask != 1] = 0

        # create birds eye view
        img = self.perspective_transform(img)
        plt.imshow(img)

        # detect lanes, and get a lane polygon img
        img = self.find_lanes_conv(img)

        # Show final lane identification
        if debug:
            plt.imshow(img)
            plt.show()

        # transform lane polygon back to normal view
        img = self.perspective_transform(img, rev = True)

        # overly lane polygon onto undistorted img
        img = cv2.addWeighted(undistort_img, 1, img, 0.3, 0)

        # Calculate the curvature
        img = self.calculate_curvature(img)

        # Show final lane overlay image
        if debug:
            plt.imshow(img)
            plt.show()

        return img

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

    def window_mask(self, width, height, img_ref, center,level):
        '''Small helper image to draw blocks over centroids given an image'''
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_lanes_conv(self, img, debug=False):
        '''Take a edge detected, perspective transformed image and detect lines
        Will update self.left and self.right with correct pixel lists, fit lines, etc.
        '''
        # Create a black/white mask
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[img != 0] = 1

        # Set some tunable values
        w_width = 50
        w_height = 80
        margin = 50

        # Pre-calculate a few common values
        levels = int(img.shape[0] / w_height)
        offset = w_width / 2

        # Initialize some values
        w_centroids = []
        w = np.ones(w_width)
        y_center = w_height * 0.5

        # Calculate the y center, bottom and center of image.
        img_bottom = int(img.shape[0] / 4 * 3)
        img_center = int(img.shape[1] / 2)

        # Sum up pixesl and calculate center for the bottom left corner
        l_sum = np.sum(img[img_bottom:, :img_center], axis=0)
        l_center = np.argmax(np.convolve(w, l_sum)) - w_width / 2

        # Sum up pixesl and calculate center for the bottom right corner
        r_sum = np.sum(img[img_bottom:, img_center:], axis=0)
        r_center = np.argmax(np.convolve(w,r_sum)) - w_width / 2 + img_center
    
        # Add all those centroids to a list
        w_centroids.append(((l_center, y_center), (r_center, y_center)))
                
        # Iterate over each level after the first, run a convolution, and calculate centroids
        for level in range(0, levels):
            # Initialize level top/bottom/center
            y_max = (level + 1) * w_height
            y_min = level * w_height
            y_center = (level + 0.5) * w_height

            # Create an image layer and convolve it
            img_layer = np.sum(img[int(img.shape[0] - y_max):int(img.shape[0] - y_min), :], axis=0)
            conv_signal = np.convolve(w, img_layer)

            # Calculate left min, max, and center
            l_min_idx = int(max(l_center + offset - margin, 0))
            l_max_idx = int(min(l_center + offset + margin, img.shape[1]))
            l_center =  np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset

            # Calculate right min, max, and center
            r_min_idx = int(max(r_center + offset - margin, 0))
            r_max_idx = int(min(r_center + offset + margin, img.shape[1]))
            r_center =  np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset

            # Upaate centroids list
            w_centroids.append(((l_center, y_center), (r_center, y_center)))

        # If no centers were found at this point, bail and return the original image 
        if len(w_centroids) <= 0:
            return img

        # Update left/right lane to reflect detected points
        self.left.points = [c[0] for c in w_centroids]
        self.right.points = [c[1] for c in w_centroids]

        # Fit a second order polynomial to right/left lane
        self.left.update_best_fit()
        self.right.update_best_fit()

        # Create an array with a values  0 to img.shape, used for poly_fit
        # XXX: and reveres it for proper fitting
        plot = np.linspace(0, img.shape[0] - 1, img.shape[0] )
        plot = plot[::-1]

        # Calculate points f(y) = Ay^2 + By + C for all y values in plot
        self.left.fit_pts  = (self.left.best_fit [0]**2) * plot + \
                self.left.best_fit [1]*plot + self.left.best_fit [2]
        self.right.fit_pts = (self.right.best_fit[0]**2) * plot + \
                self.right.best_fit[1]*plot + self.right.best_fit[2]

        # If anything goes to far left, cap it to image edge
        # XXX: not sure if we should actually do this
        self.left.fit_pts[self.left.fit_pts <  0] = 0

        # If anything goes to far right, cap it to image edge
        # XXX: not sure if we should actually do this
        self.right.fit_pts[self.right.fit_pts > img.shape[1]] = img.shape[1]

        # Call the helper function to draw a block over the original image for each found centroid
        if debug:
            l_points = np.zeros_like(img)
            r_points = np.zeros_like(img)

            for level in range(0, len(w_centroids)):
                l_mask = self.window_mask(w_width, w_height, img, w_centroids[level][0][0], level)
                l_points[l_mask == 1] = 1

                r_mask = self.window_mask(w_width, w_height, img, w_centroids[level][1][0], level)
                r_points[r_mask == 1] = 1

            img2 = np.zeros_like(img)
            img2[(l_points == 1) | (r_points == 1)] = 1
            plt.imshow(img2)
            plt.plot(self.left.fit_pts, plot, color='red')
            plt.plot(self.right.fit_pts, plot, color='green')
            plt.show()

        return self.fill_lanes(img)

    def calculate_curvature(self, img):
        '''Calculate and overkay the radius of curvature for each lane on the top corner of the image.'''
        def get_curve(y_val, poly):
            return ((1 + (2*poly[0]*y_val + poly[1])**2)**1.5) / np.absolute(2*poly[0])

        y_max = img.shape[1]
        y_val = y_max
        l_curve = get_curve(y_val, self.left.best_fit)
        r_curve = get_curve(y_val, self.right.best_fit)

        curve_txt =  "Curve radious: left %0.2f, right %0.2f" %(l_curve, r_curve)
        return cv2.putText(img, curve_txt, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))

    def fill_lanes(self, img):
        # Create a polygon that with the top/bottom points from the left/right lane
        poly_pts = [(int(self.left.fit_pts[0]), img.shape[0]),
                    (int(self.left.fit_pts[-1:]), 0),
                    (int(self.right.fit_pts[-1:]), 0),
                    (int(self.right.fit_pts[0]), img.shape[0])]

        # Draw that polygon over the original image
        img = cv2.fillPoly(img, [np.array(poly_pts)], 255)

        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img


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

    img = cv2.imread(os.path.join("test_img", "thresh_trans.jpg"))
    img = lane_lines.find_lanes_conv(img2)
    write_name = os.path.join("results", "draw-lanes.jpg")
    cv2.imwrite(write_name, transform)
    '''
    img = cv2.imread(os.path.join("test_img", "test1.jpg"))
    pipeline = lane_lines.pipeline(img, debug = True)
    write_name = os.path.join("results", "pipeline.jpg")
    cv2.imwrite(write_name, pipeline)

    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join("results", "project_video_output.mp4")
    lane_lines.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'challenge_video.mp4')
    output_vid = os.path.join("results", "challenge_video_output.mp4")
    lane_lines.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'harder_challenge_video.mp4')
    output_vid = os.path.join("results", "harder_challenge_video_output.mp4")
  
    # TODO: Get find_lanes working on the update case using classes
    # TODO: Update lane curvature to work for meters, not pixels
    # TODO: Update lane curvature to print text to screen
    # TODO: Writeup
    # TODO: Make find_lanes more robust
    # TODO: Re-test edge detection to verify find_lanes is getting decent data
    # Cleanup code
