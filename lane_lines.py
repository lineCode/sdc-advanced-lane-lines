import cv2
import numpy as np
import os
import glob
import pickle

from collections import deque

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from pipeline import Pipeline
import car_helper

class Line():
    '''Class responsible for retaining lane line information'''
    def __init__(self, side, centroids):
        # Constant pixel to meters variables
        # TODO: Calculate this dynamically
        self.ym_per_pix = 30/720 # The distance to the horizon (~30m) minues the img height (720px)
        self.xm_per_pix = 3.7/1045 # lane width (3.7m) - px difference of left/right lane (see perspective trasnform)
  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  

        # Distance from bottom pixel to center of image
        self.bottom_x = 0 # Set this to 0 instead of None, so the first few images return valid numbers

        # A fix-sized list of all centroids detected in the line
        self.points = deque(maxlen=100)

        # Side
        self.side = side

        # Number of Centroids
        self.centroid_count = centroids

        # All of the Centroid Points
        self.fit_pts = None

        # Polynomial coefficients averaged over last detected Centroid pts
        self.best_fit = None

        # The curvature of the line
        self.curve = None

        # The y_pts that map to fit_pts
        self.y_pts = None

    def update_layer1(self, points, center):
        '''Update the bottom_x value to be the calculated x at y=0'''
        self.bottom_x = self.fit_pts[-1]

    def update_points(self, centroids, idx):
        '''Given a list of centroid points and a lane index, update Line points'''
        min_pts = 50 # XXX: Tunable
        diff_thresh = 30 # XXX: Tunable
        rem_pts = 0 # XXX: Tunable
        points = centroids #[c[idx] for c in centroids]

        # If the current set of points are very different from the past n set of points, toss them
        if len(self.points) <= min_pts  or np.absolute(np.average(points) - np.average(self.points)) <= diff_thresh:
            self.points.extendleft(points)
        else:
            # In order to not get stuck in a situation where noise leads to actual changes being ignored
            # We will remove some of the older points until the new points are forced in as valid
            for i in range(0, rem_pts):
                self.points.pop()
            return -1
        return 0

    def update_best_fit(self, curve=False):
        '''Update the best fit polynomials based on current points f(y)'''
        # Extract x/y coordinates from (y, x) points and convert from pixels to meters
        x = [p[0]  for p in self.points] 
        y = [p[1] for p in self.points] 

        if curve:
            self.curve_fit = np.polyfit([n * self.ym_per_pix for n in y],
                                        [n * self.xm_per_pix for n in x], 2)
        else:
            self.best_fit = np.polyfit(y, x, 2)

    def update_fit_pts(self, y, x):
        '''Given an image.shape y max, use the polynomial fit to calculate f(y) for all values of y'''
        # Create an array with a values  0 to img.shape, used for poly_fit
        self.y_pts = np.linspace(0, y, y)

        # Calculate points f(y) = Ay^2 + By + C for all y values in plot
        self.fit_pts  = (self.best_fit[0] * self.y_pts**2) + \
                         self.best_fit[1] * self.y_pts + self.best_fit[2]

        # If anything goes to far left/right, cap it to image edge 
        # XXX: not sure if we should actually do this
        self.fit_pts[self.fit_pts <  0] = 0
        self.fit_pts[self.fit_pts > x] = x

        # Flip the points because 0 is top rather than bottom
        self.fit_pts = np.flipud(self.fit_pts)

    def update_curve(self, y_val):
        '''Given a y_val, update the line curvature'''
        # Create a best_fit in meter-space
        self.update_best_fit(curve=True)
        poly = self.curve_fit

        # Convert from pixels to meters
        y_val = y_val * self.ym_per_pix

        self.curve = ((1 + (2*poly[0]*y_val*self.ym_per_pix + poly[1])**2)**1.5) / np.absolute(2*poly[0])


class LaneLines(Pipeline):
    def __init__(self):
        super().__init__()

        # Initialize New Lanes
        self.left = Line('left', 9)
        self.right = Line('right', 9)

        # Set the perspective transform points
        # XXX: These were tuned on the test_image and assume all images 
        # XXX: are of the same viewing angle and dimensions (front-facing, 1280x720)
        # TODO: Calculate this dynamically
        x = 1280
        y = 717
        off = 50

        # XXX: Tunable
        br = [1245, 669]
        bl = [200, 675]
        tr = [730, 455]
        tl = [569, 460]

        dbr = [x - off, y - off]
        dbl = [0 + off, y - off]
        dtr = [x - off, 0 + off]
        dtl = [0 + off, 0 + off]

        self.trans_src = np.float32([bl, tl, tr, br])
        self.trans_dst = np.float32([dbl, dtl, dtr, dbr])
        self.trans_M = None
        self.trans_M_rev = None

        # Calibrate the camera and setup transformqtion matrix
        self.setup_transform()

    def setup_transform(self):
        '''Use the pre-calculated points to save the transformation matrix'''
        self.trans_M = cv2.getPerspectiveTransform(self.trans_src, self.trans_dst)
        self.trans_M_rev = cv2.getPerspectiveTransform(self.trans_dst, self.trans_src)

    def pipeline(self, img, debug=False, debug_all=False, show_centers=False):
        '''run an image through the full pipeline and return a lane-filled image
        Expects an undistorted image
        '''
        # undistort and create a copy
        undistort_img = np.copy(img)
        if debug_all:
            imgs = {'undistort': np.copy(undistort_img) }

        # create edge detection mask, and zero out anything not found in mask
        mask = self.edge_detection(img)
        img[mask != 1] = 0
        if debug_all:
            imgs['edge'] = np.copy(img)

        # create birds eye view
        img = self.perspective_transform(img)
        if debug_all:
            imgs['perspective'] = np.copy(img)

        # Display the detected lanes before img is altered in debug mode
        if debug_all or show_centers:
            img2 = self.find_lanes_conv(img, debug=True) # This will draw a debug canvas
            imgs['centers'] = np.copy(img2)
            if show_centers:
                return img2

        # detect lanes, and get a lane polygon img
        img = self.find_lanes_conv(img)
        if debug_all:
            imgs['fill'] = np.copy(img)

        # transform lane polygon back to normal view
        img = self.perspective_transform(img, rev = True)
        if debug_all:
            imgs['untransform'] = np.copy(img)

        # overly lane polygon onto undistorted img
        img = car_helper.overlay_img(undistort_img, img)

        # Calculate the curvature
        img = self.calculate_curvature(img)

        # Calculate the position of car
        img = self.calculate_car(img)
        if debug_all:
            imgs['final'] = np.copy(img)

        # Show final lane overlay image
        if debug:
            plt.imshow(img)
            plt.show()

        if debug_all:
            return imgs
        return img 

    def edge_detection(self, img):
        '''Take an <img> and return edge detected img. XXX: Tunable'''
        mask = np.zeros(img.shape[:-1])

        # Convert to HLS and take interest S values
        color_mask = self.color_thresh(img, thresh = (110, 255))
        img[color_mask != 1 ] = 0

        # Remove anything to slanted
        y_mask = self.gradient_thresh(img, orient = 'y', thresh = (30,200))

        # Remove anything to left/right leaning
        x_mask = self.gradient_thresh(img, orient = 'x',  thresh = (0,200))

        # Cut off top and sides of image
        location_mask = self.location_thresh(img, thresh = (0.6, 0.4))

        mask[(x_mask == 1) & (color_mask == 1) & 
             (y_mask == 1) & (location_mask == 1)] = 1
        return mask 

    def color_thresh(self, img, thresh=(60, 255)):
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

    def location_thresh(self, img, thresh=(0.3, 0.3)):
        '''Cut of the thresh %% top of the image'''
        mask = np.zeros(img.shape[:-1])

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

    def magnitude_thresh(self, img, thresh=(50, 120), sobel_kernel=3):
        '''Unused in final pipeline, threshold based off sobel magnitude'''
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

    def dir_thresh(self, img, thresh=(0, np.pi/2), sobel_kernel=3):
        '''Unused in final pipeline, threshold based off sobel direction'''
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

    def perspective_transform(self, img, rev=False):
        '''Transform the perspective'''
        # XXX: For some reason img shape coordinates need to be flipped here
        if rev:
            return cv2.warpPerspective(img, self.trans_M_rev, (img.shape[1], img.shape[0]), 
                    flags=cv2.INTER_LINEAR)  
        return cv2.warpPerspective(img, self.trans_M, img.shape[-2::-1], 
                flags=cv2.INTER_LINEAR)   

    def window_mask(self, width, height, img_ref, center,level):
        '''Small helper image to draw blocks over centroids given an image'''
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height) \
             : int(img_ref.shape[0] - level * height), \
               max( 0, int( center - width / 2)) \
             : min( int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def find_lanes_conv(self, img, debug=False, w_width=50, w_height=80, margin=50):
        '''Take a edge detected, perspective transformed image and detect lines
        Will update self.left and self.right with correct pixel lists, fit lines, etc.
        '''
        # Create a black/white mask
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[img != 0] = 1

        # Initialize some values
        l_centroids = []
        r_centroids = []
        w = np.ones(w_width)
        y_center = w_height * 0.5

        # Calculate the y center, bottom and center of image, and some other common values.
        levels = int(img.shape[0] / w_height)
        offset = w_width / 2
        img_bottom = int(img.shape[0] / 4 * 3)
        img_center = int(img.shape[1] / 2)

        # Sum up pixesl and calculate center for the bottom left corner
        l_sum = np.sum(img[img_bottom:, :img_center], axis=0)
        l_center = np.argmax(np.convolve(w, l_sum)) - w_width / 2

        # Sum up pixesl and calculate center for the bottom right corner
        r_sum = np.sum(img[img_bottom:, img_center:], axis=0)
        r_center = np.argmax(np.convolve(w,r_sum)) - w_width / 2 + img_center
    
        # Add all those centroids to a list
        left_pts = (l_center, y_center)
        right_pts = (r_center, y_center)
        l_centroids.append((left_pts))
        r_centroids.append((right_pts))

        # Iterate over each level after the first, run a convolution, and calculate centroids
        for level in range(0, levels):
            # TODO: Make this smarter on iterative runs
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
            l_max =  np.argmax(conv_signal[l_min_idx:l_max_idx]) 
            l_center = l_max + l_min_idx - offset

            # Calculate right min, max, and center
            r_min_idx = int(max(r_center + offset - margin, 0))
            r_max_idx = int(min(r_center + offset + margin, img.shape[1]))
            r_max =  np.argmax(conv_signal[r_min_idx:r_max_idx])
            r_center = r_max + r_min_idx - offset

            # Upaate centroids list with any found centroids
            if l_max != 0:
                l_centroids.append(((l_center, y_center)))
            if r_max != 0:
                r_centroids.append(((r_center, y_center)))

        # If no centers were found at this point, bail and return the original image 
        if len(l_centroids) and len(r_centroids) <= 0:
            return img

        # Update left/right lane to reflect detected points
        left_points = self.left.update_points(l_centroids, 0)
        right_points = self.right.update_points(r_centroids, 0)

        # Before updating any of the fit lines or fit points run sanity on the last points and all points
        sanity = self.sanity_check_lanes()

        # Update the values used to calculate car position only if the last points good
        if left_points == 0 and right_points == 0 and sanity == 0:
            # Fit a second order polynomial to right/left lane
            self.left.update_best_fit()
            self.right.update_best_fit()

            # calculate f(y) for y = [0,img.shape[0]]
            left_fit = self.left.update_fit_pts(img.shape[0], img.shape[1])
            right_fit = self.right.update_fit_pts(img.shape[0], img.shape[1])

            self.left.update_layer1(left_pts, img_center)
            self.right.update_layer1(right_pts, img_center)

        # Run sanity checks on lanes
        self.sanity_check_lanes()

        # Call the helper function to draw a block over the original image for each found centroid
        if debug:
            l_points = np.zeros_like(img)
            r_points = np.zeros_like(img)

            for level in range(0, len(l_centroids)):
                l_mask = self.window_mask(w_width, w_height, img, l_centroids[level][0], level)
                l_points[l_mask == 1] = 1

            for level in range(0, len(r_centroids)):
                r_mask = self.window_mask(w_width, w_height, img, r_centroids[level][0], level)
                r_points[r_mask == 1] = 1

            img2 = np.zeros_like(img)
            img2[(l_points == 1) | (r_points == 1)] = 1
            fig = plt.figure()
            plt.imshow(img2)

            plt.plot(self.left.fit_pts, self.left.y_pts, color='red')
            plt.plot(self.right.fit_pts, self.right.y_pts, color='green')

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return img
        return self.fill_lanes(img)

    def calculate_curvature(self, img):
        '''Calculate and overkay the radius of curvature for each lane on the top corner of the image.'''
        y_val = img.shape[0] # XXX: Currently the max, we might want to tune this
        self.left.update_curve(y_val)
        self.right.update_curve(y_val)

        curve_txt =  "Curve radius (meters): left %0.2f, right %0.2f" %(self.left.curve, self.right.curve)
        return cv2.putText(img, curve_txt, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))

    def calculate_car(self, img):
        '''Calculate the right offset of the car'''

        # pixel distance * pixel conversion. -1 gave a better value during tuning
        right_offset = ((self.right.bottom_x + self.left.bottom_x) / 2 \
                - img.shape[1] / 2) * self.left.xm_per_pix
        car_txt =  "Car is %0.3f meters right of lane center" %(right_offset)
        return cv2.putText(img, car_txt, (50, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))


    def fill_lanes(self, img):
        '''Fill in a polygon mapping to the lane location'''
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left.fit_pts, self.left.y_pts]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.fit_pts, self.right.y_pts])))])
        pts = np.hstack((pts_left, pts_right))
        pts = np.int_([pts])

        # Draw the lane onto the warped blank image in RGB
        cv2.fillPoly(img, pts, 255)
        return img

    def sanity_check_lanes(self):
        '''Run a sanity check to verify the newly detected lanes are possible'''
        # TODO: Verify lane curvature is within a reasonable margin
        # TODO: Correct a line that is out of suitable range
        # TODO: Verify they are within a reasonable horizontal range of each other
        return 0


if __name__ == '__main__':
    lane_lines = LaneLines()
    # Test calibration and update calibration data file
    lane_lines.calibrate(debug = True, read_cal = False)

    # Make sure calibration data is read in
    lane_lines.calibrate()

    # Read in test image
    img = cv2.imread(os.path.join("test_img", "test2.jpg"))
    write_name = os.path.join("results", "test-original.jpg")
    cv2.imwrite(write_name, img)
    
    # Test undistort
    undistort = lane_lines.correct_distortion(img)
    write_name = os.path.join("results", "test-undistort.jpg")
    cv2.imwrite(write_name, undistort)

    # Test color thresh
    color_img = np.copy(undistort)
    color_mask = lane_lines.color_thresh(img, thresh=(90, 255))
    color_img[color_mask != 1 ] = 0
    cv2.imwrite(os.path.join("results", "color_edge.jpg"), color_img)
    
    # Test y gradient
    y_mask = lane_lines.gradient_thresh(color_img, orient = 'y', thresh=(30,200))
    cv2.imwrite(os.path.join("results", "y_edge.jpg"), 255*y_mask)

    # Test x gradient
    x_mask = lane_lines.gradient_thresh(color_img, orient = 'x', thresh=(0,200))
    cv2.imwrite(os.path.join("results", "x_edge.jpg"), 255*x_mask)

    # Test location thresh
    loc_mask = lane_lines.location_thresh(color_img)    
    cv2.imwrite(os.path.join("results", "loc_thresh.jpg"), 255*loc_mask)

    ############ Unused in final pipeline
    # Test dir thresh
    dir_mask = lane_lines.dir_thresh(img)
    cv2.imwrite(os.path.join("results", "dir_edge.jpg"), 255*dir_mask)

    # Test magnitude thresh
    magnitude_mask = lane_lines.magnitude_thresh(img)
    cv2.imwrite(os.path.join("results", "magnitude_edge.jpg"), 255*magnitude_mask)
    ############ Unused in final pipeline

    # Test edge detection pipeline
    edge_img = np.copy(undistort)
    edge_mask = lane_lines.edge_detection(undistort)
    cv2.imwrite(os.path.join("results", "edge.jpg"), 255*edge_mask)
    edge_img[edge_mask != 1] = 0
 
    # Test perspective transofrm in-step
    transform = lane_lines.perspective_transform(edge_img)
    write_name = os.path.join("results", "transform.jpg")
    cv2.imwrite(write_name, transform)

    # Test perspective transform
    undistort_transform = lane_lines.perspective_transform(img)
    write_name = os.path.join("results", "undistort_transform.jpg")
    cv2.imwrite(write_name, undistort_transform)

    # Read in a less interesting image of a straight line to showcase transform
    img_straight = cv2.imread(os.path.join("test_img", "straight_lines1.jpg"))
    write_name = os.path.join("results", "test-straight.jpg")
    cv2.imwrite(write_name, img_straight)
    undistort_straight = lane_lines.correct_distortion(img_straight)
    edge_img_straight = np.copy(undistort_straight)
    edge_mask_straight = lane_lines.edge_detection(undistort_straight)
    edge_img_straight[edge_mask_straight != 1] = 0

    # Test perspective transform in-step on the straight line image
    transform_straight = lane_lines.perspective_transform(edge_img_straight)
    write_name = os.path.join("results", "transform_straight.jpg")
    cv2.imwrite(write_name, transform_straight)

    # Test perspective transform
    undistort_transform_straight = lane_lines.perspective_transform(lane_lines.correct_distortion(img_straight))
    write_name = os.path.join("results", "undistort_transform_straight.jpg")
    cv2.imwrite(write_name, undistort_transform_straight)

    # Test Lane detection
    lanes_img = lane_lines.find_lanes_conv(transform, debug = True)
    write_name = os.path.join("results", "lanes_lines.jpg")
    cv2.imwrite(write_name, lanes_img)

    # Test lane filling
    lanes_filled = lane_lines.find_lanes_conv(transform)
    write_name = os.path.join("results", "lanes_filled.jpg")
    cv2.imwrite(write_name, lanes_filled)

    # Test Untransform
    tranformed_lanes = lane_lines.perspective_transform(lanes_filled, rev = True)
    write_name = os.path.join("results", "transform_lanes.jpg")
    cv2.imwrite(write_name, tranformed_lanes)

    # Test curvature detection
    curve = car_helper.overlay_img(img, tranformed_lanes)
    curve_img = lane_lines.calculate_curvature(curve)
    write_name = os.path.join("results", "curves.jpg")
    cv2.imwrite(write_name, curve_img)

    # Test car detection
    car = lane_lines.calculate_curvature(curve_img)
    write_name = os.path.join("results", "car.jpg")
    cv2.imwrite(write_name, car)

    # Test full pipeline
    pipeline = lane_lines.pipeline(img)
    write_name = os.path.join("results", "pipeline.jpg")
    cv2.imwrite(write_name, pipeline)
