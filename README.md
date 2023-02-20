# Case studies Team 1

# What are we trying to do?
- Read license plate using image preprocessing techniques. Ideally - achieve high accuracy with much shorter time than using NN. 
- Read license plates using neural networks. The accuracy should be potentually higher than first method.
- Build a model that can identify nerve structures in a dataset of ultrasound images of the neck accurately.

# License plates: dataset generating

1. To generate more license plates, use:

   from license_plates.generate.plates_generate import create_plates

   create_plates(NUMBER_OF_PLATES)
   
   To change the size of output images/rotation angle range/font size, edit the function itself, parameters are hardcoded inside.

2. To generate more characters (for character recognition in MIP approach), use:

   from license_plates.generate.characters_generate import generate_characters

   generate_characters(NUM_OF_IMAGES_PER_CHARACTER)

   To change the size of output images/rotation angle range/shift parameters, edit the function itself, parameters are hardcoded inside.


# Task 1 (License plates: MIP approach)

To predict the text of the license plate on the image, use:

from license_plates.plates_read import detect_lp

img_with_contours, predicted_text = detect_lp(image)


# Task 2 (License plates: NN approach)

The total pipeline is not completed.


# Task 3 (Ultrasound nerve segmentation)

The total pipeline is not completed. 

Segmentation models in the folder models/ are not finalized vversions (the best versions were overwritten by the later less successful attempts to adjust parameters and re-train).


# Reference links and repos - Starting Point (Task 3)

- https://medium.com/analytics-vidhya/ultrasound-nerve-segmentation-an-end-to-end-image-segmentation-case-study-ec88bfed0894

- https://github.com/EdwardTyantov/ultrasound-nerve-segmentation

- https://github.com/jocicmarko/ultrasound-nerve-segmentation/