import os
import sys
import numpy as np
import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure

def main(source_dir):
    folder_name = os.path.basename(os.path.normpath(source_dir))
    destination_dir = "dataset/processed_images/"
    
    if folder_name == "normal":
        destination_dir = destination_dir + "f1"
    elif folder_name == "covid":
        destination_dir = destination_dir + "f2"
        
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    
    img_list=os.listdir(source_dir)
    
    for img_name in img_list:
        img=io.imread(os.path.join(source_dir,img_name))
        img_gray = rgb2gray(img)
        img_resized = resize(img_gray, (590,600))
        img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))
        img_enhanced=exposure.equalize_adapthist(img_rescaled)
        img_resized_8bit=img_as_ubyte(img_enhanced)
        img_name = img_name[:-3] + "bmp"
        print(img_name)
        io.imsave(os.path.join(destination_dir,img_name),img_resized_8bit)
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
                sys.exit("Use: example.py <dir>")

    main(sys.argv[1])