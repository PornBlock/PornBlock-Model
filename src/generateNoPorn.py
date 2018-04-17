"""
Contains code for randomly sampling 10 (n) images from each category
"""

import os
from random import shuffle
from shutil import copyfile

n = 1 # Images of each category

for dirpath, dirnames, filenames in os.walk("101_ObjectCategories/"):
    if len(filenames) != 0:
        # Get 10 random images
        shuffle(filenames) 
        sourceImagesPath = [ dirpath+"/" + x for x in filenames[:n]]

        # Get destination images path
        ## Get category name
        imgCategory = dirpath.split('/')[-1]
        destinationImagesPath = ["NoPorn/" + imgCategory + "_" + x for x in filenames[:n]]
    
        for i in range(len(sourceImagesPath)):
            copyfile(sourceImagesPath[i], destinationImagesPath[i])
