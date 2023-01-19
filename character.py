from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import random
import os


if os.path.exists("characters") is False:
    os.mkdir("characters")


for num in range(0,10):

    for i in range(1000):
        img = Image.new('RGB',(17,17), color = (255,255,255))      
        d = ImageDraw.Draw(img)

        fontsize = random.randint(10,20)
        font = ImageFont.truetype("EuroPlate.ttf",fontsize)

        d.text((0,0), str(num) , font=font, fill=(0, 0, 0), align="center")
        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255,255,255))
        img = img.resize((20,20))

        name = str(num) + "_" + str(i) + ".png"
        img.save("characters/"+name)
        

for let in string.ascii_uppercase:

    for i in range(1000):
        img = Image.new('RGB',(17,17), color = (255,255,255))
        d = ImageDraw.Draw(img)

        fontsize = random.randint(10,20)
        font = ImageFont.truetype("EuroPlate.ttf",fontsize)

        d.text((0,0), let , font=font, fill=(0, 0, 0), align="center")
        img = img.rotate(random.randint(-45, 45), expand=True, fillcolor=(255,255,255))
        img = img.resize((20,20))

        name = let + "_" + str(i) + ".png"
        img.save("characters/"+name)

