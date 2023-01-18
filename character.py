from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import os


if os.path.exists("characters") is False:
    os.mkdir("characters")

'''def random_color():
    rand = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())'''

plate_font = ImageFont.truetype("EuroPlate.ttf")

for num in range(0,10):

    for i in range(1000):
        #background = random_color()        
        img = Image.new('RGB',(6,9), color = (255,255,255))        #for white color = (255,255,255)
        d = ImageDraw.Draw(img)

        d.text((0,0), str(num) , font=plate_font, fill=(0, 0, 0), align="center")
        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255,255,255))
        img = ImageOps.expand(img, border=random.randint(5, 20), fill=(255,255,255))

        name = str(num) + "_" + str(i) + ".png"
        img.save("characters/"+name)

for let in string.ascii_uppercase:

    for i in range(1000):
        #background = random_color()
        img = Image.new('RGB',(6,9), color = (255,255,255))
        d = ImageDraw.Draw(img)

        d.text((0,0), let , font=plate_font, fill=(0, 0, 0), align="center")
        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255,255,255))
        img = ImageOps.expand(img, border=random.randint(5, 20), fill=(255,255,255))

        name = let + "_" + str(i) + ".png"
        img.save("characters/"+name)

