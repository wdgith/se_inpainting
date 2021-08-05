import os
import argparse
import numpy as np
import re

print("start1")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
print("start2")

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))
images.sort(key=natural_keys)
image = images[17000:20000:1]
#images = sorted(images)[16999:19999:1]  #[0:12000:20]
print(len(image))
np.savetxt(args.output, image, fmt='%s')



