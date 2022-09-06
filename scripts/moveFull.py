
import os
import cv2
import shutil
from distutils.dir_util import copy_tree

direc = "SOURCE PATH"

#directories = ([x[0] for x in os.walk(direc)])

for j in next(os.walk(direc))[1]:
    if ("Copy" in j) :
        for copyFol in ["occlusion_annotation","image_color","amodal_annotation","depth"]:
            print(direc + "\\" + j)
            currentDirec = direc + "\\" + j
            source_dir = currentDirec + "\\" + copyFol
            destination_dir = "DESTINATION PATH" + copyFol
            destination_dir_files = [subfile.split("_")[0] for subfile in os.listdir(destination_dir)]
            for file in os.listdir(source_dir):
                if(file.split("_")[0] in destination_dir_files):
                    print("double occurence",file)
                else:
                    shutil.copy2(source_dir+"\\"+file, destination_dir)