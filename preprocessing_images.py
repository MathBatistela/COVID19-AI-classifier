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
    destination_dir = os.getcwd() + "\\dataset\\preprocessed_images\\" + folder_name
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    
    image_list=os.listdir(source_dir)
    
    for img_name in image_list:
        img=io.imread(os.path.join(source_dir,img_name))
        img_gray = rgb2gray(img)
        img_resized = resize(img_gray, (600,590))#convert image size to 512*512
        img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))#min-max normalization 
        img_enhanced=exposure.equalize_adapthist(img_rescaled)#adapt hist
        img_resized_8bit=img_as_ubyte(img_enhanced)
        # img_name = "exposure" + img_name
        io.imsave(os.path.join(destination_dir,img_name),img_resized_8bit)#save enhanced image to destination dir
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
                sys.exit("Use: example.py <dir>")

    main(sys.argv[1])