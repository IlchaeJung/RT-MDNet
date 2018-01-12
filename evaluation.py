import os
from os.path import join, isdir
from tracker import *
import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0,'./modules')
from utils import *
import pickle

otb_seq_attributes = dict()
otb_seq_attributes['IV']= ['Basketball', 'Box', 'Car1', 'Car2', 'Car24', 'Car4', 'CarDark', 'Coke', 'Crowds', 'David', \
                          'Doll', 'FaceOcc2', 'Fish', 'Human2', 'Human4', 'Human7', 'Human8', 'Human9', 'Ironman', 'KiteSurf', \
                          'Lemming', 'Liquor', 'Man', 'Matrix', 'Mhyang', 'MotorRolling', 'Shaking', 'Singer1', 'Singer2', \
                          'Skating1', 'Skiing', 'Soccer', 'Sylvester', 'Tiger1', 'Tiger2', 'Trans', 'Trellis', 'Woman']
otb_seq_attributes['SV'] = ['Biker', 'BlurBody', 'BlurCar2', 'BlurOwl', 'Board', 'Box', 'Boy', 'Car1', 'Car24', 'Car4', \
                           'CarScale', 'ClifBar', 'Couple', 'Crossing', 'Dancer', 'David', 'Diving', 'Dog', 'Dog1', 'Doll',\
                           'DragonBaby', 'Dudek', 'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', \
                           'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman', 'Jump', \
                           'Lemming', 'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1',\
                           'Skater', 'Skater2', 'Skating1', 'Skating2.1', 'Skating2.2', 'Skiing', 'Soccer', 'Surfer', 'Toy', \
                           'Trans', 'Trellis', 'Twinnings', 'Vase', 'Walking', 'Walking2', 'Woman']
otb_seq_attributes['OCC'] = ['Basketball', 'Biker', 'Bird2', 'Bolt', 'Box', 'CarScale', 'ClifBar', 'Coke', 'Coupon', 'David',\
                             'David3', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Football', 'Freeman4', 'Girl', \
                             'Girl2', 'Human3', 'Human4.2', 'Human5', 'Human6', 'Human7', 'Ironman', 'Jogging.1', 'Jogging.2', \
                             'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Panda', 'RedTeam', 'Rubik', 'Singer1', 'Skating1', \
                             'Skating2.1', 'Skating2.2', 'Soccer', 'Subway', 'Suv', 'Tiger1', 'Tiger2', 'Trans', 'Walking', \
                             'Walking2', 'Woman']
otb_seq_attributes['DEF'] = [ 'Basketball', 'Bird1', 'Bird2', 'BlurBody', 'Bolt', 'Bolt2', 'Couple', 'Crossing', 'Crowds', \
                              'Dancer', 'Dancer2', 'David', 'David3', 'Diving', 'Dog', 'Dudek', 'FleetFace', 'Girl2', 'Gym', \
                              'Human3', 'Human4.2', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Jogging.1', 'Jogging.2', \
                              'Jump', 'Mhyang', 'Panda', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2.1', 'Skating2.2', \
                              'Skiing', 'Subway', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Woman']
otb_seq_attributes['MB'] =  ['Biker', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl', \
                             'Board', 'Box', 'Boy', 'ClifBar', 'David', 'Deer', 'DragonBaby', 'FleetFace', 'Girl2', 'Human2',\
                             'Human7', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'MotorRolling', 'Soccer', 'Tiger1',\
                             'Tiger2', 'Woman']
otb_seq_attributes['FM'] = ['Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', \
                            'BlurOwl', 'Board', 'Boy', 'CarScale', 'ClifBar', 'Coke', 'Couple', 'Deer', 'DragonBaby', 'Dudek', \
                            'FleetFace', 'Human6', 'Human7', 'Human9', 'Ironman', 'Jumping', 'Lemming', 'Liquor', 'Matrix', \
                            'MotorRolling', 'Skater2', 'Skating2.1', 'Skating2.2', 'Soccer', 'Surfer', 'Tiger1', 'Tiger2', \
                            'Toy', 'Vase', 'Woman']
otb_seq_attributes['IPR']=['Bird2', 'BlurBody', 'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Boy', 'CarScale', 'ClifBar', 'Coke', \
                           'Dancer', 'David', 'David2', 'Deer', 'Diving', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', \
                           'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Gym', 'Ironman', \
                           'Jump', 'KiteSurf', 'Matrix', 'MotorRolling', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', \
                           'Singer2', 'Skater', 'Skater2', 'Skiing', 'Soccer', 'Surfer', 'Suv', 'Sylvester', 'Tiger1', \
                           'Tiger2', 'Toy', 'Trellis', 'Vase']
otb_seq_attributes['OPR']=['Basketball', 'Biker', 'Bird2', 'Board', 'Bolt', 'Box', 'Boy', 'CarScale', 'Coke', 'Couple', \
                           'Dancer', 'David', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', \
                           'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', \
                           'Human2', 'Human3', 'Human6', 'Ironman', 'Jogging.1', 'Jogging.2', 'Jump', 'KiteSurf', 'Lemming', \
                           'Liquor', 'Matrix', 'Mhyang', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', \
                           'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2.1', 'Skating2.2', 'Skiing', 'Soccer', 'Surfer', \
                           'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trellis', 'Twinnings', 'Woman']
otb_seq_attributes['OV'] = ['Biker', 'Bird1', 'Board', 'Box', 'ClifBar', 'DragonBaby', 'Dudek', 'Human6', 'Ironman', \
                            'Lemming', 'Liquor', 'Panda', 'Suv', 'Tiger2']
otb_seq_attributes['BC']= ['Basketball', 'Board', 'Bolt2', 'Box', 'Car1', 'Car2', 'Car24', 'CarDark', 'ClifBar', 'Couple', \
                           'Coupon', 'Crossing', 'Crowds', 'David3', 'Deer', 'Dudek', 'Football', 'Football1', 'Human3', \
                           'Ironman', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer2',\
                           'Skating1', 'Soccer', 'Subway', 'Trellis']
otb_seq_attributes['LR'] = ['Biker', 'Car1', 'Freeman3', 'Freeman4', 'Panda', 'RedTeam', 'Skiing', 'Surfer', 'Walking']


def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type is 'OTB':
        ############################################  have to refine #############################################

        img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])

        if (seqname == 'Jogging') or (seqname == 'Skating2'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.1.txt')
        elif seqname =='Human4':
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname == 'BlurBody')  or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
                or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
                or (seqname == 'Box')   or (seqname == 'Car4')  or (seqname == 'CarScale') or (seqname == 'ClifBar') \
                or (seqname == 'Couple')  or (seqname == 'Crossing')  or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
                or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman')   :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        elif (seqname == 'Freeman4') or (seqname == 'Diving') or (seqname =='Freeman3') or (seqname =='Football1'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect_ilchae.txt', delimiter=',')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if seqname == 'David':
            img_list = img_list[300:]
            # gt = gt[300:,:]
        if seqname == 'Football1':
            img_list = img_list[0:73]
        if seqname == 'Freeman3':
            img_list = img_list[0:459]
        if seqname == 'Freeman4':
            img_list = img_list[0:282]

    elif set_type=='VOT/2016':
        img_list = sorted([seq_path + '/'+p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


if __name__ == "__main__":

    ##pick dataset
    # set_type='VOT/2016'
    set_type='OTB'
    ## path initialization
    dataset_path = '/home/ilchae/dataset/tracking/'
    seq_home = dataset_path + set_type
    ##get argument


    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]

    # bb_result = np.load('20171115_tmp2_align_otb_bb_x2_pad14_triple_result.pth.npy')
    result = np.load('tmp.pth.npy')
    result = result.tolist()

    bb_result = result['bb_result']
    success_ratio = np.zeros((100,1))

    success_ratio_att = dict()
    success_ratio_att['IV'] = np.zeros((100, 1))
    success_ratio_att['SV'] = np.zeros((100,1))
    success_ratio_att['OCC'] = np.zeros((100, 1))
    success_ratio_att['DEF'] = np.zeros((100, 1))
    success_ratio_att['MB'] = np.zeros((100, 1))
    success_ratio_att['FM'] = np.zeros((100,1))
    success_ratio_att['IPR'] = np.zeros((100, 1))
    success_ratio_att['OPR'] = np.zeros((100, 1))
    success_ratio_att['OV'] = np.zeros((100, 1))
    success_ratio_att['BC'] = np.zeros((100,1))
    success_ratio_att['LR'] = np.zeros((100,1))

    for att in ['IV','SV','OCC','DEF','MB','FM','IPR','OPR','OV','BC','LR']:
        for num, seq in enumerate(otb_seq_attributes[att]):
            try:
                if num < -1:
                    continue
                seq_path = seq_home + '/' + seq
                img_list, gt = genConfig(seq_path, set_type)

                iou_result = np.zeros((len(gt), 1))
                for i in range(0, len(bb_result[seq])):
                    iou_result[i] = overlap_ratio(bb_result[seq][i], gt[i])

                for i in range(1, success_ratio.shape[0]):
                    success_ratio_att[att][i] += ((iou_result >= (0.01*i)).astype('float32').mean()) / len(otb_seq_attributes[att])
            except:
                print 'no seq {}'.format(seq)
        success_ratio_att[att]=success_ratio_att[att].sum() / 99.

        print '{} complete'.format(att)
    print('attribute complete')

    for num,seq in enumerate(seq_list):
        if num<-1:
            continue
        seq_path = seq_home + '/' + seq
        img_list,gt=genConfig(seq_path,set_type)

        iou_result = np.zeros((len(gt),1))
        for i in range(0,len(bb_result[seq])):
            iou_result[i] = overlap_ratio(bb_result[seq][i],gt[i])

        for i in range(1,success_ratio.shape[0]):
            success_ratio[i] += ((iou_result>=0.01*(i)).astype('float32').mean())/len(seq_list)

        print '{} complete'.format(seq)


    AUC = success_ratio.sum() / 99.

    plt.plot(np.arange(0.0,1,0.01),success_ratio,'ro')
    plt.axis([0,1,0,1])
    plt.show()

    print 'finish AUC:{}'.format(AUC)

    # success_ratio /=len(seq_list)

    # MotoRolling, CarDark, Tiger2, Bird1, Human3,Skater, Singer2, Coke, Freeman4,Matrix, Bolt2, Panda, Coupon, Ironmeen, Human5,FleetFace, Diving,Jump, Rubik, BlurOwl, Skating2, FaceOcc2, Human6, Vase,, Biker, Couple, Skating1, ClifBar, Twinnings, Gym, Suv, Box, Human4, Skater2, Bolt, Dancer, BlurCar3
