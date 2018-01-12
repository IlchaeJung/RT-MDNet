import os
import numpy as np
import pickle
from collections import OrderedDict

import xml.etree.ElementTree
import xmltodict
import numpy as np

import  matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

# seq_home = '../dataset/'
# seqlist_path = '../vot-otb.txt'
# output_path = './imagenet_arrange.pkl'

output_path = './imagenet_refine.pkl'



#with open(seqlist_path,'r') as fp:
#    seq_list = fp.read().splitlines()

seq_home = '/home/ilchae/dataset/ILSVRC/'
train_list = [p for p in os.listdir(seq_home + 'Data/VID/train')]
seq_list = []
for num, cur_dir in enumerate(train_list):
    seq_list += [cur_dir + '/' + p for p in os.listdir(seq_home + 'Data/VID/train/' + cur_dir)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

data = {}
completeNum = 0
for i,seqname in enumerate(seq_list):
    print(seqname)
    seq_path = seq_home + 'Data/VID/train/' + seqname
    gt_path = seq_home +'Annotations/VID/train/' + seqname
    img_list = sorted([p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.JPEG'])

    gt = np.zeros((len(img_list),4))
    gt_list = sorted([gt_path + '/' + p for p in os.listdir(gt_path) if os.path.splitext(p)[1] == '.xml'])
    save_enable = True
    for gidx in range(0,len(img_list)):
        with open(gt_list[gidx]) as fd:
            doc = xmltodict.parse(fd.read())
        try:
            try:
                object =doc['annotation']['object'][0]
            except:
                object = doc['annotation']['object']
        except:
            ## no object, occlusion and hidden etc.
            save_enable = False
        if (int(object['trackid']) is not 0):
            save_enable = False
            break
        xmin = float(object['bndbox']['xmin'])
        xmax = float(object['bndbox']['xmax'])
        ymin = float(object['bndbox']['ymin'])
        ymax = float(object['bndbox']['ymax'])

        ## discard too big object
        if (float(doc['annotation']['size']['width'])/2. < (xmax-xmin) ) and (float(doc['annotation']['size']['height'])>2. < (ymax-ymin) ): 
            save_enable = False
            break
        gt[gidx,0] = xmin
        gt[gidx,1] = ymin
        gt[gidx,2] = xmax - xmin
        gt[gidx,3] = ymax - ymin


        # ax.cla()
        # img = np.array(Image.open(seq_path + '/'+img_list[gidx]), dtype=np.uint8)
        # ax.imshow(img)
        # box = patches.Rectangle(gt[gidx,0:2],gt[gidx,2],gt[gidx,3], linewidth=1, edgecolor ='r', facecolor = 'none')
        # ax.add_patch(box)
        # plt.draw()
        # plt.show()
        # time.sleep(1)


    if save_enable:
        assert len(img_list) == len(gt), "Lengths do not match!!"
        data[seqname] = {'images':img_list, 'gt':gt}
        completeNum += 1
        print 'Complete!'

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)

print 'complete {} videos'.format(completeNum)
