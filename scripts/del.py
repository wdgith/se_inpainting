import os
from PIL import Image
import glob
 
path = "/media/yang/Pytorch/liliqin/edge-connect-master"
 
paths = glob.glob(os.path.join(path, '*.png'))
print(1)
 
# 输出所有文件和文件夹
for file in paths:
    print(2)

    os.remove(file)
