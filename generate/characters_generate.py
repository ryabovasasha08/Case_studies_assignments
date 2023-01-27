from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import os


def generate_characters(num_samples_per_character):
    if os.path.exists("database/characters") is False:
        os.mkdir("database/characters")

    for num in range(0, 9):
        for i in range(num_samples_per_character):
            img = Image.new('RGB', (15, 20), color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            fontsize = 20
            font = ImageFont.truetype("generate/EuroPlate.ttf",fontsize)

            d.text((0, 0), str(num), font=font, fill=(0, 0, 0), align="center")
            img = img.rotate(random.randint(-10, 10), expand=True, fillcolor=(255, 255, 255))
            img = img.resize((15, 20))

            name = str(num) + "_" + str(i) + ".png"
            img.save("database/characters/" + name)

    for i in range(num_samples_per_character):
        img = Image.new('RGB', (15, 20), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        fontsize = 20
        font = ImageFont.truetype("generate/EuroPlate.ttf", fontsize)

        d.text((0, 0), "-", font=font, fill=(0, 0, 0), align="center")
        img = img.rotate(random.randint(-10, 10), expand=True, fillcolor=(255, 255, 255))
        img = img.resize((15, 20))

        name = "a" + "_" + str(i) + ".png"
        img.save("database/characters/" + name)

    for let in string.ascii_uppercase:

        for i in range(num_samples_per_character):
            img = Image.new('RGB', (15, 10), color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            fontsize = 20
            font = ImageFont.truetype("generate/EuroPlate.ttf", fontsize)

            d.text((0, 0), let, font=font, fill=(0, 0, 0), align="center")
            img = img.rotate(random.randint(-10, 10), expand=True, fillcolor=(255, 255, 255))
            img = img.resize((15, 20))

            name = let + "_" + str(i) + ".png"
            img.save("database/characters/" + name)
