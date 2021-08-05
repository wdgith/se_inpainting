import os
import argparse
import numpy as np

print("start1")

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
print(len(images))
images = [p for p in images if p.endswith('_labelIds.png')]
images = sorted(images)
np.savetxt(args.output, images, fmt='%s')