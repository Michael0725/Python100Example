import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML




class Finding_line_methods():
    def __init__(self,image):
        self.low_threshold = 70
        self.high_threshold = 150
        self.kernel_size = 3
        self.left_bottom=[0,540]
        self.right_bottom=[950,539]
        self.apex=[450,300]
        self.vertices = np.array([[self.left_bottom,self.right_bottom,self.apex]],np.int32)
        self.rho = 1
        self.theta = np.pi/180
        self.threshold = 50
        self.min_line_len = 20
        self.max_line_gap = 1000
        self.gray_pic = self.grayscale(image)
        self.gaussian_pic = self.gaussian_blur(self.gray_pic,self.kernel_size)
        self.canny_pic = self.canny(self.gray_pic,self.low_threshold,self.high_threshold)
        self.region_pic = self.region_of_interest(self.canny_pic,self.vertices)
        self.hough_pic = self.hough_lines(self.region_pic,self.rho,self.theta,self.threshold,self.min_line_len,self.max_line_gap)
        self.weight_pic = self.weighted_img(self.hough_pic,image)


    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def canny(self,img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self,img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


    def region_of_interest(self,img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    def weighted_img(self,img, initial_img, a=0.8, b=1., c=0.):
        return cv2.addWeighted(initial_img, a, img, b, c)
Input_path = "D:\\Udacity_Selfdrivingcar_Program\\Python_Code\\Find_Laneline\\CarND-LaneLines-P1-master\\CarND-LaneLines-P1-master\\test_images"
Output_path = "D:\\Udacity_Selfdrivingcar_Program\\Python_Code\\Find_Laneline\\CarND-LaneLines-P1-master\\CarND-LaneLines-P1-master\\test_images_output"
pics = os.listdir(Input_path)


def process_image(image):
    Find_line = Finding_line_methods(image)
    result = Find_line.weight_pic
    return  result

for element in pics:
    path_pic = Input_path+'\\'+element
    path_pic_output= Output_path+'\\'+element
    image = mpimg.imread(path_pic)
    process_image(image)
    cv2.imwrite(path_pic_output, process_image(image))




white_output = "D:\\Udacity_Selfdrivingcar_Program\\Python_Code\\Find_Laneline\\CarND-LaneLines-P1-master\\CarND-LaneLines-P1-master\\test_videos_output\\solidYellowLeft.mp4"
clip1 = VideoFileClip("D:\\Udacity_Selfdrivingcar_Program\\Python_Code\\Find_Laneline\\CarND-LaneLines-P1-master\\CarND-LaneLines-P1-master\\test_videos\\solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output,audio=False)

