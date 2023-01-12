from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random

plate_font = ImageFont.truetype("EuroPlate.ttf")

for num in range(0, 10):
    img = Image.new('RGB', (6, 9), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((0, 0), str(num), font=plate_font, fill=(0, 0, 0), align="center")

    img.save("characters/num_" + str(num) + ".png")

for let in string.ascii_uppercase:
    img = Image.new('RGB', (6, 9), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((0, 0), let, font=plate_font, fill=(0, 0, 0), align="center")

    img.save("upper_" + let + ".png")
