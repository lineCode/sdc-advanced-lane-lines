import cv2
import numpy as np
import os
import glob
import pickle

import matplotlib.pyplot as plt


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

    def pipeline(self, img):
        img = self.correct_distortion(img)
        img = perspective_transform(img)
        raise NotImplemented

    def perspective_transform(self, img):
        '''Transform the perspective'''
        # XXX: For some reason img shape coordinates need to be flipped here
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

    def find_lanes(self, binary_warped, debug=False):
        raise NotImplemented

    def fill_lanes(self, img):
        raise NotImplemented

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

if __name__ == '__main__':
    lane_lines = LaneLines()
    img = cv2.imread(os.path.join("test_img", "test1.jpg"))
    img_blank = np.zeros_like(img)
    # Test calibration and update calibration data file
    lane_lines.calibrate(debug = True, read_cal = False)

    img_blank = np.zeros_like(img)
    # Test Image Transformation
    transform_img = cv2.imread(os.path.join("test_img", "transform_test.jpg"))
    
    undistort = lane_lines.correct_distortion(transform_img)
    write_name = os.path.join("results", "test-undistort.jpg")
    cv2.imwrite(write_name, undistort)

    transform = lane_lines.perspective_transform(undistort)
    write_name = os.path.join("results", "test-transform.jpg")
    cv2.imwrite(write_name, transform)

    # Test color thresh
    color_mask = lane_lines.color_thresh(img)
    cv2.imwrite(os.path.join("results", "color_edge.jpg"), 255*color_mask)
    
    #img = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
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
    loc_img = lane_lines.location_thresh(img)    
    cv2.imwrite(os.path.join("results", "loc_thresh.jpg"), 255*loc_img)

    # Test edge detection pipelien
    edge_img = lane_lines.edge_detection(img)
    cv2.imwrite(os.path.join("results", "edge.jpg"), 255*edge_img)
    
    #lane_lines.lane_curvature(img, True)

    # TODO: Get find_lanes using the line class
    # TODO: Get find_lanes working on the update case
    # TODO: Get lane_curvature working
    # TODO: Get fill_lanes Working
    # TODO: Get pipeline working
    # TODO: Test with a video
    # TODO: Writeup