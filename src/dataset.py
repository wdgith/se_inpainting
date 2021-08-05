import os
import glob
import scipy
import torch
import random
import numpy as np
import matplotlib
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
#from scipy.misc import imread
#from cv2 import imread
from src.utils import Progbar, create_dir, stitch_images, imsave,imsave_np1
from skimage.feature import canny
from imageio import imread
#from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist,seg_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)
        self.seg_data = self.load_flist(seg_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.seg = config.SEG
        self.mask = config.MASK
        self.nms = config.NMS
        

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        #return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except Exception as e:
            print(e)
            print('loading error: ' + self.data[index])
            item = self.load_item(0)
        #print("load succ %s"%self.data[index])
        return item

    def load_name(self, index):
        image_name = self.data[index]
        #image_name = self.data[index]
        return os.path.basename(image_name)

    def load_item(self, index):

        size = self.input_size
        #size = 512

        # load image
        #img = imread(self.data[index])

        img = Image.open(self.data[index])
        img = img.resize((size, size), Image.BILINEAR)
        
        #img = np.array(img)
        

        # gray to rgb
        

        # resize/crop if needed
        #img = self.resize(img, size, size)
        

        img = np.array(img)
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # create grayscale image
        img_gray = rgb2gray(img)
        #img_gray = img.convert("L")

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img_gray, index, mask)
        #scipy.misc.imsave("edge"+os.path.basename(self.data[index]),edge)
        #imsave(im,os.path.join('edge',os.path.basename(self.data[index])))


        # load segmentation
        seg = self.load_seg(img, index, mask)

        
        
        

    
        # augment data
        
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...].copy()
            img_gray = img_gray[:, ::-1, ...].copy()
            edge = edge[:, ::-1, ...].copy()
            seg = seg[:, ::-1, ...].copy()
            mask = mask[:, ::-1, ...].copy()

        
      

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge),torch.FloatTensor(seg), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        #mask = None if self.training else (1 - mask / 255).astype(np.bool)
        mask=None

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(np.float)
            #return canny(img, sigma=sigma).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_seg(self,img, index, mask):

        #mask = None if self.training else (1 - mask / 255).astype(np.bool)

        #从img到对应segmatation
        #1 直接 2 分割算法
        if self.seg==1:
            #print("load seg directly")
            imgh, imgw = img.shape[0:2]
            seg_list=[]
            name = self.load_name(index)
            fname, fext = name.split('.')
            map = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            for i in range(18):
                path = os.path.join("/media/yang/Pytorch/liliqin/dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno",str(int(fname)//2000),fname.zfill(5)+'_'+map[i]+".png")
                #seg = imread(pathi)
                #seg = self.resize(seg, imgh, imgw)
                if os.path.exists(path):
                    seg = Image.open(path).convert('L')     
                    seg = seg.resize((imgh, imgw), Image.BILINEAR)
                    seg = np.array(seg) 
                    #seg = imread(path)
                    #seg = self.resize(seg, imgh, imgw)
                    #seg = rgb2gray(seg)
                    seg[seg < 150]=0
                    seg[seg >= 150]=1
                    

                else:
                    seg = np.zeros((imgh, imgw))
                
                
                #seg = (seg < 50).astype(np.uint8) * 0

                
                seg_list.append(seg)
            segback = np.zeros((imgh, imgw))
            for item in seg_list:
                segback +=item
            segback [segback >0] =1
            segback =1-segback
            seg_list.append(segback)
            seg_list = np.asarray(seg_list)
            #path1 = os.path.join(self.results_path, 'edge'+name)
            #plt.imshow(seg)
            #imsave(seg,'/media/yang/Pytorch/liliqin/edge-connect-master/output/seg')
            #matplotlib.image.imsave('%s'%os.path.basename(self.seg_data[index]), seg)
            #seg=seg*mask
            #seg = rgb2gray(seg)
            return seg_list
        else:
            print("load seg from deepv3+ model")
            return 0
        #

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            #mask = Image.imread(self.mask_data[mask_index])
            mask = Image.open(self.mask_data[mask_index])
            mask = mask.resize((imgh, imgw),Image.BICUBIC)
            mask = np.array(mask)

            #mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = Image.open(self.mask_data[index])
            mask = mask.resize((imgh, imgw),Image.BICUBIC)
            mask = np.array(mask)
            """
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            """
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
    """
    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, 'gtFine', phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

        image_dir = os.path.join(root, 'leftImg8bit', phase)
        image_paths = make_dataset(image_dir, recursive=True)

        if not opt.no_instance:
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
        else:
            instance_paths = []

        # load mask
        mask_dir = os.path.join(root, 'irregular_mask')
        mask_paths = make_dataset(mask_dir, recursive=True)

        return label_paths, image_paths, instance_paths, mask_paths
    """
    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])
