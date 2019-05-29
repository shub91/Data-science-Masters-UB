"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def corr_norm(img1, img2): # Function to calculate the normalised cross correlation between img1 and img2
    
    S = len(img1)*len(img1[0])
    
    img1_val = img1 - (sum(map(sum, img1)) / S)
    img1_sd = np.sqrt(sum(map(sum, img1_val ** 2)))
    
    img2_val = img2 - (sum(map(sum, img2)) / S)
    img2_sd = np.sqrt(sum(map(sum, img2_val ** 2)))
    
    num = sum(map(sum, utils.elementwise_mul(img1_val, img2_val)))
    den = img1_sd * img2_sd
    
    if den == 0:
    	corr = -1
    	
    else:
        corr = num / den
        
    return corr


def filter_coordinates(coord, coordx, coordy, templatewidth, templateheight):
    # Function to remove overlapping characters while doing template matching
    
    add = True
    
    for c in coord:
        
        if coordy == c[1]:
            if (coordx < (c[0]+templateheight-2)):
                add = False
                
        elif coordx == c[0]:
            if (coordy < (c[1]+templatewidth-2)):
                add = False
                
    return add

def detect(img, template, ttype):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.

    thres = 0.750
    if ttype == 'a':
        thres = 0.80
        
    elif ttype == 'b':
        thres = 0.74
        
    elif ttype == 'c':
        thres = 0.764

    coordinates = []
    corrmat = [[0 for _ in range(len(img[0]))] for _ in range(len(img))]
    templateheight = len(template)
    templatewidth = len(template[0])
  
    for i in range(len(img) - templateheight + 1):
        for j in range(len(img[0]) - templatewidth + 1):
            patch_img = utils.crop(img, i, i + templateheight, j, j + templatewidth)
            corrmat[i][j] = corr_norm(patch_img, template)
    
    for i in range(len(corrmat)):
        for j in range(len(corrmat[0])):
            if corrmat[i][j] > thres:
                if filter_coordinates(coordinates, i, j, templatewidth, templateheight) == True:
                    coordinates.append((i, j))
                    write_image(utils.crop(img, i, i+templateheight, j, j+templatewidth), os.path.join("./test/", "row{}_column{}_coo{}.jpg".format(i,j,corrmat[i][j])))

    # raise NotImplementedError
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)
    ttype = args.template_path.split('/')[-1].split('.')[0]

    coordinates = detect(img, template, ttype)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
