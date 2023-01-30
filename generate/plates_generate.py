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
        text += " "
    else:
        text += "-"
    for i in range(0, random.randint(1, 2)):
        text += random.choice(string.ascii_uppercase)
    text += " "
    for i in range(0, random.randint(1, 4)):
        text += str(random.randint(0, 9))
    return text


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def create_plates(N):
    if os.path.exists("database/plates") is False:
        os.mkdir("database/plates")

    if os.path.exists("database/masks") is False:
        os.mkdir("database/masks")

    for i in range(0, N):
        is_multiline = random.randint(0, 1)
        H = 50
        W = 80 if is_multiline else 160
        font = ImageFont.truetype("generate/EuroPlate.ttf", 20)

        # Create image with random light background color of size (W, H)
        img = Image.new('RGB', (W, H), color=random_color_light())
        d = ImageDraw.Draw(img)

        # Create a random plate number according to German syntax and put it in the center of image (can be 1 or 2liner)
        text = random_plate_number(is_multiline)
        text = " ".join(text)
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        d.multiline_text(((W - w) / 2 - min((W - w) / 2 - 5, random.randint(0, 15)), (H - h) / 2), text, font=font,
                         fill=(0, 0, 0), align="center")

        # Stack horizontally with a blue sign of D and EU flag, scaled to correspond to image size
        d_sign = Image.open("generate/d_sign.jpg")
        d_sign_w, d_sign_h = d_sign.size
        ratio = d_sign_w / d_sign_h
        d_sign = d_sign.resize((int(ratio * H), H), Image.Resampling.LANCZOS)
        img = concat_images_horiz(d_sign, img)

        # create mask
        mask = img.copy()
        mask.paste((255, 255, 255), [0, 0, mask.size[0], mask.size[1]])

        # Add black border to the obtained license plate
        img = ImageOps.expand(img, border=2, fill='black')

        # Rotate and pad the license plate by random padding with random background but output images have same size
        background = random_color()
        rotate_angle = random.randint(-20, 20)
        border = random.randint(1, 15)
        img = img.rotate(rotate_angle, expand=True, fillcolor=background)
        img = ImageOps.expand(img, border=border, fill=background)
        img = img.resize((160, 80))
        mask = mask.rotate(rotate_angle, expand=True, fillcolor='black')
        mask = ImageOps.expand(mask, border=border, fill='black')
        mask = mask.resize((160, 80))
        mask = expand2square(mask, 'black')
        img = expand2square(img, background)
        img = img.resize((256, 256))
        mask = mask.resize((256, 256))

        img.save("database/plates/" + re.sub(r"[\n\t\s]*", "", text) + ".png")
        mask.save("database/masks/" + re.sub(r"[\n\t\s]*", "", text) + ".png")

# create_plates(10)
