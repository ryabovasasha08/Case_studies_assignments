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


def detect_lp(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Otsu's threshold to clean the image and only leave important - works like a charm
    otsu_threshold, gray = cv2.threshold(np.uint8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tried dilation+erosion to get rid of white D letter, but not worked since it noises up the plate text too.

    # Apply Canny edge detection
    # gray = cv2.Canny(np.uint8(gray), 50, 150)
    # Find contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []
    for cnt in contours:
        # Get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(cnt)
        # Don't plot small false positives that aren't license plates
        if w > 35 or h > 35 or w < 10 or y < 10:
            continue
        bounding_rects.append([x, y, w, h])
        # Draw the rectangle around the license plate
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (150, 150, 150), 1)

    # Here we need to come up with some condition to remove D based on it being further from the rest of letters

    return gray


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

    if i > 5:
        break
