'''Abstract Pipeilne Class Module'''
import os
import glob
import pickle
import cv2
from cv2 import undistort
import numpy as np
import matplotlib.pyplot as plt

class Pipeline(object):
    '''This is an abstract pipeline class

    It is expected that some CarWorld will consume this.
    As such you should implement the pipeline(self, img) function.
    Also implement all supporting functions.

    It is also expected that image undistortion/camera calibration has been done
    before any images reach this point.
    '''
    def __init__(self):
        super().__init__()

        # Set directories
        self.cal_dir = "camera_cal"
        self.results_dir = "results"

        # Set Calibration defaults
        self.dist_mtx = None
        self.dist_dist = None
        self.cal_file = os.path.join(self.results_dir, "calibration_data.p")

    def calibrate(self, x = 9, y = 6, debug=False, read_cal=True):
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
        def undistort_images(debug, images):
            '''Save undistorted images if debug==true'''
            if debug:
                for idx, fname in enumerate(images):
                    img = cv2.imread(fname)
                    img = self.correct_distortion(img)
                    write_name = os.path.join(self.results_dir, 
                                'undistort' + str(idx) + '.jpg')
                    cv2.imwrite(write_name, img)

        # Try to read calibration data in first
        if read_cal:
            print("Reading pre-calculated calibration data")
            try:
                cal_data = pickle.load(open(self.cal_file, "rb" ))
                self.dist_mtx = cal_data['dist_mtx']
                self.dist_dist = cal_data['dist_dist']
                undistort_images(debug, images)
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

            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
            
            # If we found corners, update the lists and do some debug
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

        # save images
        undistort_images(debug, images)

        # Save data
        cal_data = {'dist_mtx': self.dist_mtx, 'dist_dist': self.dist_dist}
        dist_pickle = pickle.dump(cal_data, open(self.cal_file, "wb" ))

    def correct_distortion(self, img):
        '''Given an image, use pre-calculated correction/distortion matrices to undistort the image'''
        assert self.dist_mtx is not None
        assert self.dist_dist is not None
        assert img is not None
        undistort_img =  undistort(img, self.dist_mtx, self.dist_dist)
        assert img.shape == undistort_img.shape
        return undistort_img

    def pipeline(self, img):
        '''Return  processed image same size as input'''
        raise NotImplementedError

    def debug_pipeline(self, img):
        '''Debug wrapper for pipeline that expects a map of images in response'''
        print("entering debug pipeline")
        imgs = self.pipeline(img, debug_all=True)

        # Create a new image big enough to store all the output images
        shape = imgs['final'].shape[0:2]
        scale = len(imgs) + 1
        output = np.zeros((shape[0] * scale, shape[1], 3))

        # Iterate over each image and lay them on top of each other
        i = 1
        for img in sorted(imgs.keys()):
            # If it is grayscale make it RGB
            if len(imgs[img].shape) == 2:
                imgs[img] = cv2.cvtColor(imgs[img], cv2.COLOR_GRAY2RGB)

            # Add it to the output and increment the offset, make smaller imgs fit
            output[i * shape[0]: ( i + 1) * shape[0] - (shape[0] - imgs[img].shape[0]),
                    0:imgs[img].shape[1],:] = imgs[img]
            i += 1
        return output