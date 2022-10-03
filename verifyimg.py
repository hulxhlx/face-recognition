import os
import shutil
import argparse
import numpy as np

from PIL import Image

#os.remove(path) #Delete file
#os.removedirs(path) #Delete empty folder


def find_corrupt(folder_path):
    data_dir = folder_path
    flds = os.listdir(data_dir)

    for fld in flds:
        if os.path.isdir(data_dir + '/' + fld):
            sub_flds = os.listdir(data_dir + '/' + fld)
            for i in sub_flds:
                i_path = data_dir + '/' + fld + '/' + i
                try:
                    img = Image.open(i_path)
                    img.verify() # verify that it is, in fact an image
                except:
                    print('Bad file:', i_path)
                    os.remove(i_path) # delete bad file

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="____")
    PARSER.add_argument('-f', '--folder_path')
    ARGS = PARSER.parse_args()
    find_corrupt(str(ARGS.folder_path))