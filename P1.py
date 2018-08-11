#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

debug = 1
increase = 0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
     # return the edged image
    if(debug == 1):
        imageCopy = np.uint8(image)
    else:
        imageCopy = image
    return cv2.Canny(imageCopy, lower, upper)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def fit_lines(img, points, color, thickness):
    """
    Draw a best fit line through all the point on 1 side (left or right)
    """
    global increase
    if (len(points)):
        # given start & end of y dimension for limiting the drawing line
        start_y = int(img.shape[0])
        end_y = int(img.shape[0]/2 + 80)
        #calculate vector & point using fit line
        vx, vy, cx, cy = cv2.fitLine(np.float32(points), cv2.DIST_L2 ,0,0.01,0.01)
        # compute slope & intercept for line: y = ax +b
        slope = vy / vx
        intercept = cy - (slope * cx)
        if(math.fabs(slope) > 0.5):
            start_x = int((start_y - intercept) / slope)
            end_x = int((end_y - intercept) / slope)
            #draw line
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def filter_out_lines_by_angle(img, lines,  color=[255, 0, 0], thickness=5):
    """
    This funtion is to filter out the line base on angle
    to get the most sufficient line using atan equation

    For left -40 < angle < -30
    For right 25 < angle < 35

    """
    filtered_point_right = []
    filtered_point_left = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # calculate angle
            alpha = math.degrees(math.atan((y2-y1)/(x2-x1)))
            if(alpha < 40 and alpha > 20):
                filtered_point_right.append([x1,y1])
                filtered_point_right.append([x2,y2])
            elif (alpha < -30 and alpha > -40):
                filtered_point_left.append([x1,y1])
                filtered_point_left.append([x2,y2])
    #draw left line
    fit_lines(img, filtered_point_left, color, thickness)
    #draw right line
    fit_lines(img, filtered_point_right, color, thickness)

def filter_out_lines_by_slope(img, lines, color=[255, 0, 0], thickness=5):
    """
    This function is to filter out the line base on slope
    to get the most sufficient line base on slope (follow tips in draw_lines function)
    """
    left_points = []
    right_points = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if math.fabs(slope) > 0.5: # <-- Only consider extreme slope
                if ( slope < 0):
                    left_points.append([x1,y1])
                    left_points.append([x2,y2])
                else:
                    right_points.append([x1,y1])
                    right_points.append([x2,y2])
    # draw left line
    fit_lines(img, left_points, color, thickness)
    # draw rigimg, right_points, color, thickness)

def filter_out_lines_by_cross_product(img, lines, color=[255, 0, 0], thickness=5):
    one_side = []
    another_side = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (determine_side_by_cross_product(img, x1, y1, x2, y2) == 1):
                one_side.append([x1,y1])
                one_side.append([x2,y2])
            elif (determine_side_by_cross_product(img, x1, y1, x2, y2) == 2):
                another_side.append([x1,y1])
                another_side.append([x2,y2])
            else:
                print("Cross the line")
    # draw one line
    fit_lines( img, one_side, color, thickness)
    # draw another line
    fit_lines(img, another_side, color, thickness)

def determine_side_by_cross_product(img, x1, y1, x2, y2):
    res = 0
    #get the reference line
    x1_ref = x2_ref= img.shape[1]/2
    y1_ref = 0
    y2_ref = img.shape[0]
    xp1 = (x2_ref - x1_ref) * (y1 - y1_ref) - (y2_ref - y1_ref) * (x1 - x1_ref)
    xp2 = (x2_ref - x1_ref) * (y2 - y1_ref) - (y2_ref - y1_ref) * (x2 - x1_ref)
    if (xp1 > 0 and xp2 > 0):
        res = 1
    elif (xp1 < 0 and xp2 < 0):
        res = 2
    return res


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img_default = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    #filter_out_lines_by_slope(line_img,lines)
    filter_out_lines_by_cross_product(line_img,lines)
    #filter_out_lines_by_angle(line_img,lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def filter_color(img):
    # Grab the x and y size and make a copy of the image
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define color selection criteria
    #only care about the color of bottom half of image
    ycal = int(img.shape[0]/2)
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = np.mean(img[ycal:,:,0]) + (np.max(img[ycal:,:,0]) - np.mean(img[ycal:,:,0])) / 2
    green_threshold = np.mean(img[ycal:,:,1]) + (np.max(img[ycal:,:,1]) - np.mean(img[ycal:,:,1])) / 2
    blue_threshold = np.mean(img[ycal:,:,2]) + (np.max(img[ycal:,:,2]) - np.mean(img[ycal:,:,2])) / 2
    ######
    ## mask o white
    mask1 = cv2.inRange(img, (red_threshold,green_threshold,blue_threshold), (255,255,255))
    ## mask o yellow (15,0,0) ~ (36, 255, 255)
    mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))
    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(img,img,mask=mask)
    return result

def filter_white_color2(image):
    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    #only care about the color of bottom half of image
    ycal = int(ysize/2)

    # Define color selection criteria
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = np.mean(image[ycal:,:,0]) + (np.max(image[ycal:,:,0]) - np.mean(image[ycal:,:,0])) / 2
    green_threshold = np.mean(image[ycal:,:,1]) + (np.max(image[ycal:,:,1]) - np.mean(image[ycal:,:,1])) / 2
    blue_threshold = np.mean(image[ycal:,:,2]) + (np.max(image[ycal:,:,2]) - np.mean(image[ycal:,:,2])) / 2
    ######

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
                | (image[:,:,1] < rgb_threshold[1]) \
                | (image[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0,0,0]
    return color_select

def process_image(image):
    #gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)                   
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    filtered_image = filter_color(image)
    gray = cv2.cvtColor(filtered_image,cv2.COLOR_RGB2GRAY) 
    blur_filtered_image = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Define our parameters for Canny and apply
    edges = auto_canny(blur_filtered_image)
    # This time we are defining a four sided polygon to mask
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [0, int(ysize)]
    right_bottom = [int(xsize), int(ysize)]
    apex = [int(xsize/2), int(ysize*3/5)]

    # only look at the interested region
    vertices = np.array([[left_bottom ,apex, apex, right_bottom]], dtype=np.int32)
    masked_edges = region_of_interest(gray,vertices)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    min_line_length = 10 #minimum number of pixels making up a line
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap)
    # Draw the lines on the edge image
    result = weighted_img(image, line_image, α=0.8, β=1., γ=0.)
    return result

def test_image():
    files = os.listdir("test_images/")
    for file in files:
        image = mpimg.imread("test_images/" + file)
        plt.imshow(process_image(image))
        plt.show()

def test_video(): 
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def test_video_yellow():
    yellow_output = 'test_videos_output/solidYellowLeft.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,1)
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

def test_video_chanlenge():
    challenge_output = 'test_videos_output/challenge.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
    #clip3 = VideoFileClip('test_videos/challenge.mp4')
    challenge_clip = clip3.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

#test_image()
#test_video()
#test_video_yellow()
#test_video_chanlenge()