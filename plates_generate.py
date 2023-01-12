from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import cv2
import re
import os
import numpy as np

'''-------------GENERATE THE PLATES------------'''


def concat_images_horiz(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def random_color_light():
    rand = lambda: random.randint(220, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())


def random_color():
    rand = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())


def random_plate_number(multiline):
    text = ""
    for i in range(0, random.randint(1, 3)):
        text += random.choice(string.ascii_uppercase)
    if multiline:
        text += "\n"
    elif random.randint(0, 9) > 3:
        text += "  "
    else:
        text += "-"
    for i in range(0, random.randint(1, 2)):
        text += random.choice(string.ascii_uppercase)
    text += " "
    for i in range(0, random.randint(1, 4)):
        text += str(random.randint(0, 9))
    return text


def create_plates(N):
    for i in range(0, N):
        is_multiline = random.randint(0, 1)
        H = 50
        W = 70 if is_multiline else 140
        font = ImageFont.truetype("EuroPlate.ttf", 20)

        # Create image with random light background color of size (W, H)
        img = Image.new('RGB', (W, H), color=random_color_light())
        d = ImageDraw.Draw(img)

        # Create a random plate number according to German syntax and put it in the center of image (can be 1 or 2liner)
        text = random_plate_number(is_multiline)
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        d.multiline_text(((W - w) / 2 - min((W - w) / 2 - 5, random.randint(0, 15)), (H - h) / 2), text, font=font,
                         fill=(0, 0, 0), align="center")

        # Stack horizontally with a blue sign of D and EU flag, scaled to correspond to image size
        d_sign = Image.open("d_sign.jpg")
        d_sign_w, d_sign_h = d_sign.size
        ratio = d_sign_w / d_sign_h
        d_sign = d_sign.resize((int(ratio * H), H), Image.Resampling.LANCZOS)
        img = concat_images_horiz(d_sign, img)

        # Add black border to the obtained license plate
        img = ImageOps.expand(img, border=2, fill='black')

        # Rotate and pad the license plate by random padding with random background
        background = random_color()
        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=background)
        img = ImageOps.expand(img, border=random.randint(5, 30), fill=background)

        img.save("plates/" + re.sub(r"[\n\t\s]*", "", text) + ".png")
