from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import cv2
import re
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
import string
import metrics
import random
import cv2
import re
import os
import numpy as np
from matplotlib import cm
from generate.characters_generate import generate_characters
from generate.plates_generate import create_plates
from read_classic.train_model_characters import create_and_train_model, convert_to_text
from utils import load_bw_images_dict_from_folder

'''--------------READ THE PLATES----------------'''


def get_center(rect):
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)


def split_in_two_lines(rects, deviation):
    for first_line_first_index in range(0, len(rects)):
        for first_line_second_index in range(0, len(rects)):
            if first_line_first_index != first_line_second_index:
                first_line_rect_indices = [first_line_first_index, first_line_second_index]
                line_1_rects = [rects[first_line_rect_indices[0]], rects[first_line_rect_indices[1]]]
                second_line_rect_indices = []
                line_2_rects = []

                point1 = get_center(line_1_rects[0])
                point2 = get_center(line_1_rects[1])
                eps = 10 ^ (-5)

                line_1_m = (point2[1] - point1[1]) / (point2[0] - point1[0] + eps)
                line_1_b = point1[1] - (line_1_m * point1[0])
                line_2_m = 0
                line_2_b = 0

                are_two_parallel_lines = True

                for i in range(0, len(rects)):
                    center = get_center(rects[i])

                    if i in first_line_rect_indices or i in second_line_rect_indices:
                        continue
                    else:
                        line_1_y = line_1_m * center[0] + line_1_b
                        line_2_y = line_2_m * center[0] + line_2_b

                        if abs(center[1] - line_1_y) < deviation:
                            line_1_rects.append(rects[i])
                            first_line_rect_indices.append(i)
                        elif len(line_2_rects) > 1:
                            if abs(center[1] - line_2_y) < deviation:
                                line_2_rects.append(rects[i])
                                second_line_rect_indices.append(i)
                            else:
                                are_two_parallel_lines = False
                                break
                        else:
                            line_2_rects.append(rects[i])
                            second_line_rect_indices.append(i)
                            point = get_center(line_2_rects[0])
                            line_2_m = line_1_m
                            line_2_b = point[1] - (line_2_m * point[0])

                if are_two_parallel_lines:
                    if line_1_b < line_2_b:
                        return line_1_rects, line_2_rects
                    else:
                        return line_2_rects, line_1_rects


def get_box_points(x, y, w, h, alpha):
    return cv2.boxPoints(((x, y), (w, h), alpha))


# Fix the wrong formula (now rects are rotated, so formula for x2, y2 is wrong)
def remove_overlapping_bounding_rects(bounding_rects, overlap_percent):
    overlapped_indices = []
    max_idx = len(bounding_rects)
    for chosen_rect_idx in range(max_idx):
        x1, y1, h1, w1, alpha1 = bounding_rects[chosen_rect_idx]
        area1 = w1 * h1
        for j in range(chosen_rect_idx + 1, max_idx):
            if chosen_rect_idx != j:
                x2, y2, h2, w2, alpha2 = bounding_rects[j]
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                area_overlap = x_overlap * y_overlap
                area2 = w2 * h2
                if area_overlap > overlap_percent * min(area1, area2):
                    if area1 < area2:
                        overlapped_indices.append(chosen_rect_idx)
                    else:
                        overlapped_indices.append(j)
    k = len(bounding_rects)
    overlapped_indices = list(dict.fromkeys(overlapped_indices))
    np_output = np.array(bounding_rects)
    np_output = np.delete(np_output, overlapped_indices, 0)
    return np_output.tolist()


# Get a segment of size (15, 20) - the same size as characters in database
def get_square_segment(x, y, w, h, alpha, size, gray, i):
    box = get_box_points(x, y, w, h, alpha)
    box = np.int0(box)
    # print(box)
    # the order of the box points: first the lowest one, and then clockwise from there.
    # So it can be: bottom left, top left, top right, bottom right
    # Or it can be: bottom right, bottom left, top left, top right
    # Check which case is that now:
    # print(alpha)
    if alpha >= 45:
        last_box_point = box[3].copy()
        box[3] = box[2]
        box[2] = box[1]
        box[1] = box[0]
        box[0] = last_box_point
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(gray, M, (int(w), int(h)))
    warped = Image.fromarray(warped, mode="L")
    # warped = ImageOps.expand(warped, border=5, fill='white')
    warped = warped.resize((size - 5, size))
    warped = np.array(warped)

    # cv2.imshow("warped" + str(i), warped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return warped


# 9000 is empirical value - only 8 sometines doesn't satisfy
def is_segment_dash(img):
    print(ndimage.variance(img))
    return ndimage.variance(img) < 9000


def detect_lp(gray, text):
    # Try dilation-erosion to make bigger whitespaces between letters - didn't work
    # kernel = np.ones((2, 2), np.uint8)
    # Using cv2.dilate() method
    # gray = cv2.dilate(gray, kernel)

    # Tried dilation+erosion to get rid of white D letter, but not worked since it noises up the plate text too.

    # Apply Canny edge detection
    # gray = cv2.Canny(np.uint8(gray), 50, 150)

    '''Find contours in the image'''
    # cv2.findContours finds arbitrary template in the grayscale image using Generalized Hough Transform
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []
    for cnt in contours:
        # Get rectangle bounding contour
        (x, y), (h, w), alpha = cv2.minAreaRect(cnt)

        # Don't plot small false positives that aren't license plates
        # Height can be really small though (in case of dash)
        if w > 35 or h > 35 or w < 8 or h < 5:
            continue
        bounding_rects.append([x, y, h, w, alpha])

    '''Removing letter D as the most left rectangle and all the defined rectangles that overlap with it'''
    most_left_rect = min(bounding_rects, key=lambda t: t[0])
    bounding_rects.remove(most_left_rect)
    chosen_x1, chosen_y1, chosen_h, chosen_w, chosen_alpha = most_left_rect
    # Fix the wrong formula (now rects are rotated, so formula for x2, y2 is wrong)
    chosen_x2, chosen_y2 = chosen_x1 + chosen_w, chosen_y1 + chosen_h
    for rect in bounding_rects:
        x1, y1, h, w, alpha = rect
        x2, y2 = x1 + w, y1 + h
        if (x1 < chosen_x2 and x2 > chosen_x1 and
                y1 < chosen_y2 and y2 > chosen_y1):
            bounding_rects.remove(rect)

    # print(len(bounding_rects))
    '''Remove overlapping bounding rectangles'''
    bounding_rects = remove_overlapping_bounding_rects(bounding_rects, 0.5)
    # print(len(bounding_rects))

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

    ''' Convert all the segments in the separate images of the correct size (15*20 - same as chars in database)'''
    img_list = [get_square_segment(x, y, h, w, alpha, 20, gray, i) for i, [x, y, h, w, alpha] in
                enumerate(bounding_rects)]

    predicted_text = convert_to_text(img_list)

    '''If the bounding rectangle is too wide and short- it is probably a dash'''
    for i, [x, y, h, w, alpha] in enumerate(bounding_rects):
        if w > h * 2 and i != 0 and i != len(bounding_rects) - 1:
            predicted_text[i] = "-"

    '''Try check variance of segment to recognize dashes'''
    # for i, img in enumerate(img_list):
    #     if is_segment_dash(img):
    #         predicted_text[i] = "-"

    rgb_image = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    '''Draw the rectangles around the characters in license plate'''
    for [x, y, h, w, alpha] in bounding_rects:
        box = get_box_points(x, y, h, w, alpha)
        box = np.int0(box)
        cv2.drawContours(rgb_image, [box], 0, (0, 0, 255), 1)

    return rgb_image, "".join(predicted_text)


# Number of plates (right now in 'plates' folder there's 7k images
N = 100
#generate_characters(N)
# create_plates(N)  # to generate a folder 'plates' with N images and masks with N images
# model = create_and_train_model()

images_dict = load_bw_images_dict_from_folder("database/plates")
images = list(images_dict.values())
images_platenames = list(images_dict.keys())

test_size = 100
avg_percent_of_correct_placed_chars = 0
avg_length_identical = 0
avg_text_identical = 0

for i, image in enumerate(images):
    img_with_contours, predicted_text = detect_lp(image, images_platenames[i])
    text = images_platenames[i][0:images_platenames[i].find('.')]

    print("Image number ", i)
    print("Real text: ", text)
    print("Predicted text: ", predicted_text)

    avg_percent_of_correct_placed_chars += metrics.get_correct_placed_chars_percent(text, predicted_text)
    avg_length_identical += metrics.is_length_same(text, predicted_text)
    avg_text_identical +=  metrics.is_text_same(text, predicted_text)

    # cv2.imshow(images_platenames[i], img_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    if i > 100:
        break

avg_percent_of_correct_placed_chars /= test_size
avg_length_identical /= test_size
avg_text_identical /= test_size
print("avg_percent_of_correct_placed_chars: ", avg_percent_of_correct_placed_chars)
print("avg_length_identical: ", avg_length_identical)
print("avg_text_identical: ", avg_text_identical)