# **Finding Lane Lines on the Road**

---
**About Me:**

- Name: Duc Le :)
- Working as embedded engineer for automotive and completely new to computer vision.
- Very much excited about Autonomous Car Driving topic
- Hope that I would have an great time learning

HAPPY LEARNING !

---
**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
---
[//]: # (Image References)
[image1]: ./write_up_image/Auto_Canny.png "Auto Canny"
[image2]: ./write_up_image/filtered_out_by_color.png "Color filter"
[image3]: ./write_up_image/extract_roi.png "Extract ROI"
[image4]: ./write_up_image/result_image.png "Result Image"
### Reflection

### A. Pipeline design

The my pipeline design consists 7 step:
- Filter out the white color and yellow color
- Convert to gray scale
- Using Gaussian Blur to ignore noise
- Using Canny to detect edge
- Extract the ROI (region of interest)
- Detect Line by using HoughLine function
- Connect all the line segment together to get the final output

Let's take a look closer to each of the step:
#### 1. Filter out the white color and yellow color
This step I have added after completed the project and trying to improve the accuracy of pipline. I realized that I forgot something in the lecture and look back all the video then I got a hint that lane line can be easily filtered out by color, either white or yellow (at least with the video condition it is true :)
To filter out the yellow color, first I convert the image to HSV color system and then pick the sufficient value for _low_threshold_ and _high_threshold_ (check the color space [here](https://alloyui.com/examples/color-picker/hsv))
```python
    ## mask o yellow (15,0,0) ~ (36, 255, 255)
    mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))
```
I got a big problem with white color, I could not find the best fit value for white in HSV color space to filter out. In order to obtain the best result, I have use the RGB system as shown in lecture video :) - nothing could be improve here at the moment
```python
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = np.mean(img[ycal:,:,0]) + (np.max(img[ycal:,:,0]) - np.mean(img[ycal:,:,0])) / 2
    green_threshold = np.mean(img[ycal:,:,1]) + (np.max(img[ycal:,:,1]) - np.mean(img[ycal:,:,1])) / 2
    blue_threshold = np.mean(img[ycal:,:,2]) + (np.max(img[ycal:,:,2]) - np.mean(img[ycal:,:,2])) / 2
    ######
    ## mask o white
    mask1 = cv2.inRange(img, (red_threshold,green_threshold,blue_threshold), (255,255,255))
```
Final step is to combine both mask 1 (for white) and mask 2 (for yellow) by using bit_wise_or and maks out the image by using bit_wise_and.
```python
    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(img,img,mask=mask)
```
![Color Filter][image2]
#### 2. Convert Gray Scale & using Gaussian Blur to ignore noise:
- This can be done quite simple by calling
    ```python
    gray = cv2.cvtColor(filtered_image,cv2.COLOR_RGB2GRAY)
    blur_filtered_image = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    ```
- I have also tried to use morphologyEx for removing noise and smoothen the image, but there is no significant improvement compared to Gaussian Blur. Futhermore, coding for GaussianBlur looks quite simple so that I keep the exsiting implementation as it is.

#### 3. Using Canny for edge detection:
Next step is to feed the image through Canny function for edge detection, there are some parametter that need to be tuned to make it works properly for this project as shown in lecture video, the good value should be as following
``` python
    lowThreshold = 50
    highThreshold = 150
```
But I think that it should be automatically tuned somehow by using the knowledge of pixel value, I found the great article [here](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/) for the auto_canny. The result is exactly as expected :)
```python
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
     # return the edged image
    imageCopy = image
    return cv2.Canny(imageCopy, lower, upper)
```

Following is the output of auto_canny
![Auto Canny][image1]

#### 4. Extract ROI:
This function is to extract the ROI to reduce the ambiguity and complexity for our algorithm to detect the lane line. Currently for the first project I have hardcoded the verticies as following:
```python
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [0, int(ysize)]
    right_bottom = [int(xsize), int(ysize)]
    apex = [int(xsize/2), int(ysize*3/5)]
    # only look at the interested region
    vertices = np.array([[left_bottom ,apex, apex, right_bottom]], dtype=np.int32)
    masked_edges = region_of_interest(gray,vertices)
```

The implementation of region_of_interest is quite simple by using fillPoly with ignoring mask color as 255
```python
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
```
![Extract ROI][image3]
#### 5.Detect Line by using HoughLine function:
Next step is to detect all the line using HoughLine function, I refer to use HoughLineP to get all of the line, so that it could be easier for processing draw_line() function. The ussage of HoughLineP is quite simple but again, to meet the expectation of this project, I have to tune a lot of parametter by using _trial and false_ approach :) And after some tries, this set of parametter seem to be working for me:
```python
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    min_line_length = 10 #minimum number of pixels making up a line
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
```
The most challenging for me is how to identify the left and right lane to draw a big line through them. I have use 3 approach:
- Identify by using line angle
- Identify by using line slope
- Identify by using reference line

First two approaches have similar computation since they are all depend on calculating slope.
- With the first approach, I compute degree by ```tan^-1((y2-y1)/(x2-x1))``` and filtering out the unsufficient value.
- On second apporach, I compute the slope value by ```slope = (y2-y1)/(x2-x1)``` and filtering out base on the sign of slope:
 ```
    if ( slope < 0):
        left_points.append([x1,y1])
        left_points.append([x2,y2])
    else:
        right_points.append([x1,y1])
        right_points.append([x2,y2])
```

Eventhough both of them working fine for me for all the test image and test video, but while trying for challenging video, I could see the accuracy is very bad eventhough I only consider the extreme slope (``` if math.fabs(slope) > 0.5: # <-- Only consider extreme slope```) As trying for continous improvement, I have use the last approach by using the concept of cross product. I use the vertical line at the center of image as reference line and compute the cross product of each output line from HoughLineP. The reference for cross product is [here](https://en.wikipedia.org/wiki/Cross_product)
**Reference code:**
```python
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
```

I could not really tell which one is better since all of the parametter is specially tuned for fiting this project only, but for now I could see the result of last approach is slightly better on the chanlenging video :)

#### 6. Connect all the segment together and compose final output
I have done this by using cv2.fitLine to find the best fit line through all of the left point and right point.
```python
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
```
![Result][image4]
### 2. Identify potential shortcomings with your current pipeline

There are a lot of shortcomings for my current pipeline.

**1. Accuracy is not as expected:** I could see it is not working for chalenging video, this is because of the street color at certain point of time it is al most close to white color and too much noise even with Gaussian Blur could not completely removed.

**2. ROI is fixed region:** Currently the ROI is fixed as shown earlier which is not adaptable. In real world, ROI should change base on the road condition. For ex: if the car is climbing up the hill then the ROI should be reduced. In contrast, if car is going down hill then the ROI should be extended. Furthermore the shape of ROI should not be consistant also, if car is running on a left curve then the shape of ROI should be somehow more narrow on the left side and wider on the right side.

**3.Line segemented as straight line:** This is also a big shortcomings for this pipeline, since the land line could be a curve not always a straight line.

**4.Not consider about weather condition:** This pipeline will completely not working in case of bad weather like rainy or snowy day.

**5. Not detect other vehicle:**: This pipeline could not detect the vehicle on the road, since it could also block the land line while crossing. Furthermore this pipline could not extrapolate the land line base on the previous data and the car driving direction.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be as following:

**1. Accuracy is not as expected:** In order to solve this issue, I am thinking of some approaches:
- Using morphologyEx with MORPH_CLOSE with hoping to eliminate the noise
- Increase the contrast and/or find the most optimum value to filter out the white line
- Also need more experiment to find the best tuning value for exsiting pipeline

**2. ROI is fixed region:** An adaptive ROI has been proposed [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1031.3880&rep=rep1&type=pdf) which could be apply for this project to see the improvement. The solution is to initiate a fix ROI from begining then finding the vanishing point to calculate the new ROI and update to the pipline.

**3.Line segemented as straight line:**: As of now I do not have any idea how to solve this issue. Maybe I will try to plot all the point and find the contour through them, but as I am completely new to computer vision so I will spend some more time later to find out the solution for this.

**4 Not consider about weather condition:** Well, this is a big topic actually. I could see there is a lot of research trying to improve the detection accuracy even using Machine Learning algorithm. I am not yet decided which approaches could be use for a this project at all. Again, I need more time diving in this topic as I have very less experiences

**5. Not detect other vehicle:**: Looking at the curriculum of this course, I could see this will be discussed in some next session :) Until then I do not have any idea yet :)
