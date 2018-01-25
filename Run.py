import os
from os.path import join, isdir
from tracker import *
import numpy as np

import argparse

import pickle




def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'OTB':
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'OTB' )
    parser.add_argument("-model_path", default = './models/imagenet-vgg-m.mat')
    parser.add_argument("-padding_ratio", default = 5., type = float)
    parser.add_argument("-result_path", default = './result.pth')
    parser.add_argument("-profiling_enable",default=False, action = 'store_true')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-short_update_enable",default=True, action = 'store_false')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=False, action='store_true')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')
    parser.add_argument("-online_jitter", default=False, action='store_true')
    parser.add_argument("-maxiter_update", default=15, type=int)
    parser.add_argument("-multi_scale_infer", default = False, action = 'store_true')

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['padding_ratio'] = args.padding_ratio
    opts['padded_img_size'] = opts['img_size']*int(opts['padding_ratio'])
    opts['result_path']=args.result_path
    opts['profiling_enable']=args.profiling_enable
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['short_update_enable'] = args.short_update_enable
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    opts['online_jitter'] = args.online_jitter
    opts['maxiter_update'] = args.maxiter_update
    opts['multi_scale_infer'] = args.multi_scale_infer
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print opts


    ##pick dataset
    # set_type='VOT/2016'
    ## path initialization
    dataset_path = '/home/ilchae/dataset/tracking/'
    seq_home = dataset_path + opts['set_type']
    ##get argument

    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()
    for num,seq in enumerate(seq_list):
        if num<-1:
            continue
        seq_path = seq_home + '/' + seq
        img_list,gt=genConfig(seq_path,opts['set_type'])

        # opts['visual_log']=True
        # if (seq == 'Box') or (seq == 'Basketball') or (seq == 'Biker'):
        # if (seq == 'BlurOwl'):
        # if (seq == 'Coke') or (seq == 'Sylvester')  :
        #     opts['visualize']=True
        iou_result, result_bb, fps = run_mdnet(img_list, gt[0], gt, display=opts['visualize'])

        iou_list.append(iou_result.mean())
        bb_result[seq] = result_bb
        fps_list[seq]=fps
        #
        ###### David must check!!
        print '{} : {} , total mIoU:{}, fps:{}'.format(seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))

    result['bb_result']=bb_result
    result['fps']=fps_list
    # np.save('iou_result.pth',iou_list)
    np.save(opts['result_path'],result)

    # MotoRolling, CarDark, Tiger2, Bird1, Human3,Skater, Singer2, Coke, Freeman4,Matrix, Bolt2, Panda, Coupon, Ironmeen, Human5, FleetFace, Diving,Jump, Rubik, BlurOwl, Skating2, FaceOcc2, Human6, Vase,, Biker, Couple, Skating1, ClifBar, Twinnings, Gym, Suv, Box, Human4, Skater2, Bolt, Dancer, BlurCar3
    # MotoRolling, CarDark, Bird1, Skater, Singer2, Coke, Freeman4,Matrix, Bolt2, Panda, Coupon, Ironmeen, Human5, FleetFace, Diving,Jump, Rubik, BlurOwl, Skating2, FaceOcc2, Human6, Vase,, Biker, Couple, Skating1, ClifBar, Twinnings, Gym, Suv, Box, Human4, Skater2, Bolt, Dancer, BlurCar3
