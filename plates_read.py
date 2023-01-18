from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import cv2
import re
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import cv2
import re
import os
import numpy as np

'''--------------READ THE PLATES----------------'''


def load_images_dict_from_folder(folder):
    images_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
        if img is not None:
            images_dict[filename] = img
    return images_dict


def get_center(rect):
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)


def split_in_two_lines(rects, deviation):
    while True:
        first_line_rect_indices = random.sample(range(len(rects)), 2)
        line_1_rects = [rects[first_line_rect_indices[0]], rects[first_line_rect_indices[1]]]
        line_2_rects = []

        point1 = get_center(line_1_rects[0])
        point2 = get_center(line_1_rects[1])

        line_1_m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        line_1_b = point1[1] - (line_1_m * point1[0])
        line_2_m = 0
        line_2_b = 0

        are_two_parallel_lines = True

        for i in range(1, len(rects)):
            center = get_center(rects[i])

            if i == first_line_rect_indices[0] or i == first_line_rect_indices[1]:
                continue
            else:
                line_1_y = line_1_m * center[0] + line_1_b
                line_2_y = line_2_m * center[0] + line_2_b

                if abs(center[1] - line_1_y) < deviation:
                    line_1_rects.append(rects[i])
                elif len(line_2_rects) >= 1:
                    if abs(center[1] - line_2_y) < deviation:
                        line_2_rects.append(rects[i])
                    else:
                        are_two_parallel_lines = False
                        break
                else:
                    line_2_rects.append(rects[i])
                    point = get_center(line_2_rects[0])
                    line_2_m = line_1_m
                    line_2_b = point[1] - (line_2_m * point[0])

        if are_two_parallel_lines:
            if line_1_b < line_2_b:
                return line_1_rects, line_2_rects
            else:
                return line_2_rects, line_1_rects


def detect_lp(image):
    '''Convert to grayscale'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''Use Otsu's threshold to clean the image and only leave important - works like a charm'''
    otsu_threshold, gray = cv2.threshold(np.uint8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tried dilation+erosion to get rid of white D letter, but not worked since it noises up the plate text too.

    # Apply Canny edge detection
    # gray = cv2.Canny(np.uint8(gray), 50, 150)

    '''Find contours in the image'''
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []
    for cnt in contours:
        # Get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(cnt)
        # Don't plot small false positives that aren't license plates
        if w > 35 or h > 35 or w < 10 or y < 10:
            continue
        bounding_rects.append([x, y, w, h])

    '''Removing letter D as the most left rectangle and all the defined rectangles that overlap with it'''
    most_left_rect = min(bounding_rects, key=lambda t: t[0])
    bounding_rects.remove(most_left_rect)
    chosen_x1, chosen_y1, chosen_w, chosen_h = most_left_rect
    chosen_x2, chosen_y2 = chosen_x1 + chosen_w, chosen_y1 + chosen_h
    for rect in bounding_rects:
        x1, y1, w, h = rect
        x2, y2 = x1 + w, y1 + h
        if (x1 < chosen_x2 and x2 > chosen_x1 and
                y1 < chosen_y2 and y2 > chosen_y1):
            bounding_rects.remove(rect)

    '''Check if the license plate is 1-liner or 2-liners'''
    is_one_line = True
    bounding_rects_centers = []
    for rect in bounding_rects:
        bounding_rects_centers.append(get_center(rect))
    deviation = 5
    # Fit a line to the points using the numpy polyfit function
    line_coefs = np.polyfit([point[0] for point in bounding_rects_centers],
                            [point[1] for point in bounding_rects_centers], 1)

    # Calculate the y-coordinates for the line at each x-coordinate
    line_y_values = [line_coefs[0] * x + line_coefs[1] for x, _ in bounding_rects_centers]

    # Check the deviation of each point from the line
    for point, line_y in zip(bounding_rects_centers, line_y_values):
        if abs(point[1] - line_y) > deviation:
            is_one_line = False

    # print(is_one_line)

    '''Sort the segments in the correct order'''
    if is_one_line:
        bounding_rects.sort(key=lambda r: r[0])
    else:
        first_line, second_line = split_in_two_lines(bounding_rects, deviation)
        first_line.sort(key=lambda r: r[0])
        second_line.sort(key=lambda r: r[0])
        first_line.extend(second_line)
        bounding_rects = first_line

    '''HERE TAKE SEQUENCE BOUNDING_RECTS AND RECOGNIZE SYMBOLS FROM IT - TBD'''

    # Draw the rectangles around the characters in license plate
    for [x, y, w, h] in bounding_rects:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (150, 150, 150), 1)

    return bounding_rects


# Number of plates (right now in 'plates' folder there's 7k images
N = 7000
# create_plates(N) # to generate a folder 'plates' with N images

images_dict = load_images_dict_from_folder("plates")
images = list(images_dict.values())
images_platenames = list(images_dict.keys())
for i, image in enumerate(images):
    cv2.imshow(images_platenames[i], detect_lp(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if i > 30:
        break
