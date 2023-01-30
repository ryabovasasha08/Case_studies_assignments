from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import os
import numpy as np


def generate_characters(num_samples_per_character):
    if os.path.exists("database/characters") is False:
        os.mkdir("database/characters")

    for num in range(0, 10):
        for i in range(num_samples_per_character):
            W = 10
            H = 16
            img = Image.new('RGB', (W, H), color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            fontsize = 20
            font = ImageFont.truetype("generate/EuroPlate.ttf", fontsize)

            _, _, w, h = d.textbbox((0, 0), str(num), font=font)

            d.text(((W - w) / 2, (H - h) / 2), str(num), font=font, fill=(0, 0, 0),
                   align="center")

            top_padding = random.randint(0, 1)
            left_padding = random.randint(0, 1)
            right_padding = random.randint(0, 1)
            bottom_padding = random.randint(0, 1)

            img = np.array(img)
            img = img[top_padding:h - top_padding - bottom_padding, left_padding:w - left_padding - right_padding]
            img = Image.fromarray(img)

            img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255, 255, 255))


            img = img.resize((15, 20))

            name = str(num) + "_" + str(i) + ".png"
            img.save("database/characters/" + name)

    for i in range(num_samples_per_character):
        W = 10
        H = 16
        img = Image.new('RGB', (W, H), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        fontsize = 20
        font = ImageFont.truetype("generate/EuroPlate.ttf", fontsize)
        _, _, w, h = d.textbbox((0, 0), "-", font=font)
        d.text(((W - w) / 2, (H - h) / 2), "-", font=font, fill=(0, 0, 0), align="center")

        top_padding = random.randint(0, 1)
        left_padding = random.randint(0, 1)
        right_padding = random.randint(0, 1)
        bottom_padding = random.randint(0, 1)

        img = np.array(img)
        img = img[top_padding:h - top_padding - bottom_padding, left_padding:w - left_padding - right_padding]
        img = Image.fromarray(img)

        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255, 255, 255))

        img = img.resize((15, 20))

        name = "AA" + "_" + str(i) + ".png"
        img.save("database/characters/" + name)

    for let in string.ascii_uppercase:
        for i in range(num_samples_per_character):
            W = 10
            H = 16
            img = Image.new('RGB', (W, H), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            _, _, w, h = d.textbbox((0, 0), let, font=font)
            fontsize = 20
            font = ImageFont.truetype("generate/EuroPlate.ttf", fontsize)

            d.text(((W - w) / 2, (H - h) / 2), let, font=font, fill=(0, 0, 0), align="center")

            top_padding = random.randint(0, 1)
            left_padding = random.randint(0, 1)
            right_padding = random.randint(0, 1)
            bottom_padding = random.randint(0, 1)

            img = np.array(img)
            img = img[top_padding:h - top_padding - bottom_padding, left_padding:w - left_padding - right_padding]
            img = Image.fromarray(img)

            img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255, 255, 255))

            img = img.resize((15, 20))

            name = let + "_" + str(i) + ".png"
            img.save("database/characters/" + name)
