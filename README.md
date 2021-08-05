# SEinpaint

![Alt text](https://github.com/wdgith/se_inpainting/blob/master/module.svg)

## config.yaml
Option          | Description
----------------| -----------
MODE: 1         |   # 1: train, 2: test, 3: eval
MODEL: 1        |   # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
MASK: 3         |   # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
EDGE: 1         |   # 1: canny, 2: external
SEG: 1          |   # 1:direct, 2: external
NMS: 1          |   # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
SEED: 10        |   # random seed
GPU: [0,1]      |   # list of gpu ids  single or multi
DEBUG: 1        |   # turns on debugging mode
VERBOSE: 0      |   # turns on verbose mode in the output console

Option          | Description
----------------| -----------
TRAIN_FLIST:                    |       /media/yang/Pytorch/liliqin/celeba/datasets/train.flist
TRAIN_SEG_FLIST: 
TRAIN_MASK_FLIST:               |       /media/yang/Pytorch/liliqin/celeba/datasets/mask.flist
TEST_FLIST:                     |       /media/yang/Pytorch/liliqin/celeba/datasets/test3000.flist
TEST_MASK_FLIST:                |       /media/yang/Pytorch/liliqin/celeba/datasets/test_mask3000.flist
TEST_SEG_FLIST:                 |       /media/yang/Pytorch/liliqin/edge-connect-master/datasets/test_seg1500.flist

Option                | Description
----------------------|------------
LR: 0.0001            |        # learning rate
ES: 0.0001            |        # edge module learning rate
DS: 0.1               |        # SPDISC  learning rate
DE: 0.1               |       # EPDISC  learning rate
D2G_LR: 0.1           |       # discriminator/generator learning rate ratio
BETA1: 0.0            |       # adam optimizer beta1
BETA2: 0.9            |       # adam optimizer beta2
BATCH_SIZE: 8         |      # input batch size for training
INPUT_SIZE: 256       |       # input image size for training 0 for original size
SIGMA: 2              |       # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
MAX_ITERS: 2e6        |       # maximum number of iterations to train the model

Option                       | Description
-----------------------------|------------
S_FM_LOSS_WEIGHT:            |  20
S_ADV_LOSS_WEIGHT:           |  1
E_FM_LOSS_WEIGHT:            |  3
E_ADV_LOSS_WEIGHT:           |  1
EDGE_THRESHOLD: 0.5          |    # edge detection threshold
L1_LOSS_WEIGHT: 1            |    # l1 loss weight
FM_LOSS_WEIGHT: 10           |    # feature-matching loss weight
STYLE_LOSS_WEIGHT: 250       |    # style loss weight
CONTENT_LOSS_WEIGHT: 0.1     |    # perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT: 0.1 |    # adversarial loss weight
GAN_LOSS: nsgan              |     # nsgan | lsgan | hinge
GAN_POOL_SIZE: 0             |     # fake images pool size
SAVE_INTERVAL: 1000          |     # how many iterations to wait before saving model (0: never)
SAMPLE_INTERVAL: 1000        |     # how many iterations to wait before sampling (0: never)
SAMPLE_SIZE: 12              |     # number of images to sample
EVAL_INTERVAL: 0             |     # how many iterations to wait before model evaluation (0: never)
LOG_INTERVAL: 10             |     # how many iterations to wait before logging training status (0: never)

## train 
python train.py --model 1  --path [checkpoint_path]  
   
## test
python test.py --model 1 --path [checkpoint_path]
