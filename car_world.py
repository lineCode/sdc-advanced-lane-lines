
'''The CarWorld Class it the main class for the car project.
This class will 
  process the input (video or image)
  run the data through various detection pipelines (lane/vehicle detection)
  return annoted output
'''
from moviepy.editor import VideoFileClip
import os

import lane_lines
# import vehicle_detection
import car_helper

from pipeline import Pipeline

class CarWorld(Pipeline):
    def __init__(self):
        super().__init__()
        self.calibrate() # Calibration cannot occur until after super init
        self.lanes = lane_lines.LaneLines()
        # self.vehicles = vehicle_detection.VehicleDetection(False, "big")

    def process_video(self, input_vid, output_vid, debug=False):
        '''Run an video through the pipeline, allow debug options'''
        print("Running %s through pipeline and outputting to %s" %(input_vid, output_vid))
        clip = VideoFileClip(input_vid)
        func = self.pipeline
        if debug:
            keys = ['img', 'undistort', 'edge', 'perspective', 'centers', 'fill', 'untransform', 'final']
            func = self.debug_pipeline
        output = clip.fl_image(func)
        output.write_videofile(output_vid, audio = False)

    def pipeline(self, img, debug_all=False):
        '''Run an image through the pipeline
        pipline is an overlay of lane detection and vehicle detection
        '''
        # Correct for camera distortion
        img = self.correct_distortion(img)

        # Identify Lanes
        lanes_img = self.lanes.pipeline(img, debug_all=debug_all)
        if not debug_all:
            assert lanes_img.shape == img.shape

        # Identify Cars
        # vehicles_img = self.vehicles.pipeline(img)
        # assert vehicles_img.shape == img.shape

        # Combine Results
        
        # img = car_helper.overlay_img(lanes_img, vehicles_img)
        # img = car_helper.overlay_img(img, lanes_img)

        return lanes_img

if __name__ == '__main__':
    cw = CarWorld()

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join("results", "project_video_output.mp4")
    cw.process_video(input_vid, output_vid)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join("results", "project_video_output_debug.mp4")
    cw.process_video(input_vid, output_vid, debug=True)

    input_vid = os.path.join("test_vid",'challenge_video.mp4')
    output_vid = os.path.join("results", "challenge_video_output.mp4")
    cw.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'harder_challenge_video.mp4')
    output_vid = os.path.join("results", "harder_challenge_video_output.mp4")
    cw.process_video(input_vid, output_vid)
  