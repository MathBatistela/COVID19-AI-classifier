import os
import sys
import numpy as np
import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure

def main(source_dir):
    
    destination_dir = "data/processed_images/"    
    data = {
        "normal": { "source": source_dir + "/normal", "destination": destination_dir + "f1"  },
        "covid": { "source": source_dir + "/covid", "destination": destination_dir + "f2"  }
    }
    
    for folder in data.keys():
        
        if not os.path.exists(data[folder]['destination']):
            os.mkdir(data[folder]['destination'])            
        
        for img_name in os.listdir(data[folder]['source']):
            img=io.imread(os.path.join(data[folder]['source'],img_name))
            img_gray = rgb2gray(img)
            img_resized = resize(img_gray, (590,600))
            img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))
            img_enhanced=exposure.equalize_adapthist(img_rescaled)
            img_resized_8bit=img_as_ubyte(img_enhanced)
            img_name = img_name[:-3] + "bmp"
            print(img_name)
            io.imsave(os.path.join(data[folder]['destination'],img_name),img_resized_8bit)        

if __name__ == "__main__":
    if len(sys.argv) != 2:
                sys.exit("Use: example.py <dir>")

    main(sys.argv[1])