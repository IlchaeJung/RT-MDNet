import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'./modules')
from data_prov import *
from model import *
from pretrain_options import *

import numpy as np

import argparse


## for validation, opts is used to track the target in validation set
from tracker import *



# print img_home, data_path

#i set_type='VOT/2015'
## path initialization
# data_path = '/home/ilchae/dataset/tracking/'


def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'], w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

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


def train_mdnet():

    ## set image directory
    if pretrain_opts['set_type'] == 'OTB':
        img_home = '/home/ilchae/dataset/tracking/OTB/'
        data_path = './otb-vot15.pkl'
    if pretrain_opts['set_type'] == 'VOT':
        img_home = '/home/ilchae/dataset/tracking/VOT/'
        data_path = './vot-otb.pkl'
    if pretrain_opts['set_type'] == 'IMAGENET':
        img_home = '/home/ilchae/dataset/ILSVRC/Data/VID/train/'
        # data_path = './imagenet.pkl'
        # data_path = './modules/imagenet_arrange.pkl'
        data_path = './modules/imagenet_refine.pkl'

    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)



    K = len(data)

    ## Init model ##
    model = MDNet(pretrain_opts['init_model_path'], K)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)

        opts['adaptive_align']=True
        opts['jitter'] = True
        opts['online_jitter']=False

    if pretrain_opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(pretrain_opts['ft_layers'])
    model.train()

    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt = seq['gt']
        if pretrain_opts['set_type'] == 'OTB':
            img_dir = os.path.join(img_home, seqname+'/img')
        if pretrain_opts['set_type'] == 'VOT':
            img_dir = img_home + seqname
        if pretrain_opts['set_type'] == 'IMAGENET':
            img_dir = img_home + seqname
        # dataset[k] = RegionDataset(img_dir, img_list, gt, model.receptive_field, pretrain_opts)
        if pretrain_opts['efficient_data_prov']:
            dataset[k]=RegionDataset4TripleNetEfficient(img_dir,img_list,gt,model.receptive_field,pretrain_opts)
        else:
            dataset[k] = RegionDataset4TripleNet(img_dir, img_list, gt, model.receptive_field, pretrain_opts)


    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    tripleCriterion = TripleLoss(0.5)
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])

    best_score = 0.
    batch_cur_idx = 0
    for i in range(pretrain_opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        totalTripleLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            try:
                cropped_scenes, pos_rois, neg_rois= dataset[k].next()
            except:
                continue

            if pretrain_opts['efficient_data_prov']:
                try:
                    for sidx in range(0, len(cropped_scenes)):
                        cur_scene = cropped_scenes[sidx]
                        cur_pos_rois = pos_rois[sidx]
                        cur_neg_rois = neg_rois[sidx]

                        cur_scene = Variable(cur_scene)
                        cur_pos_rois = Variable(cur_pos_rois)
                        cur_neg_rois = Variable(cur_neg_rois)
                        if pretrain_opts['use_gpu']:
                            cur_scene = cur_scene.cuda()
                            cur_pos_rois = cur_pos_rois.cuda()
                            cur_neg_rois = cur_neg_rois.cuda()
                        cur_feat_map = model(cur_scene, k, out_layer='conv3')

                        cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                        cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                        cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                        cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                        if sidx == 0:
                            pos_feats = [cur_pos_feats]
                            neg_feats = [cur_neg_feats]
                        else:
                            pos_feats.append(cur_pos_feats)
                            neg_feats.append(cur_neg_feats)
                    feat_dim = cur_neg_feats.size(1)
                    pos_feats = torch.stack(pos_feats,dim=0).view(-1,feat_dim)
                    neg_feats = torch.stack(neg_feats,dim=0).view(-1,feat_dim)

                except:
                    continue
            else:
                cropped_scenes= Variable(cropped_scenes)
                pos_rois = Variable(pos_rois)
                neg_rois = Variable(neg_rois)
                if pretrain_opts['use_gpu']:
                    cropped_scenes = cropped_scenes.cuda()
                    pos_rois = pos_rois.cuda()
                    neg_rois = neg_rois.cuda()
                feat_maps = model(cropped_scenes,k,out_layer='conv3')
                try:
                    pos_feats = model.roi_align_model(feat_maps,pos_rois)
                    pos_feats = pos_feats.view(pos_feats.size(0), -1)
                    neg_feats = model.roi_align_model(feat_maps,neg_rois)
                    neg_feats = neg_feats.view(neg_feats.size(0), -1)
                except:
                    continue


            pos_score = model(pos_feats, k, in_layer='fc4')
            neg_score = model(neg_feats, k, in_layer='fc4')

            cls_loss = binaryCriterion(pos_score, neg_score)


            if pretrain_opts['triple_loss_enable']:
            ## ======================= prepare triplet losses =================================
                ## anchor features generation
                aidx = Variable(torch.from_numpy(np.random.permutation(pos_feats.size(0)))).cuda()
                anchor_feats = pos_feats.index_select(0,aidx)
                # anchor_rois = pos_rois.index_select(0, aidx)
                # anchor_feats = model.roi_align_model(feat_maps,anchor_rois)
                # anchor_feats = anchor_feats.view(anchor_feats.size(0),-1)

                ## hard neg features generation
                _, top_idx = neg_score.data[:,1].topk(anchor_feats.size(0))
                hard_neg_feats = neg_feats.index_select(0,Variable(top_idx))
                # hard_neg_rois = neg_rois.index_select(0, Variable(top_idx))
                # hard_neg_feats = model.roi_align_model(feat_maps, hard_neg_rois)
                # hard_neg_feats = hard_neg_feats.view(hard_neg_feats.size(0),-1)

                triple_loss = tripleCriterion(anchor_feats, pos_feats, hard_neg_feats)

                if ((i/50)%2)==0:
                    cls_w = 1.
                    trp_w = 0.01
                else:
                    cls_w = 0.01
                    trp_w = 1.0
                cls_w = 1.
                trp_w = 1.
                (cls_w*cls_loss+trp_w*triple_loss).backward()
            else:
                cls_loss.backward()

            batch_cur_idx+=1
            if (batch_cur_idx%pretrain_opts['seqbatch_size'])==0:
                torch.nn.utils.clip_grad_norm(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0

            ## evaulator
            prec[k] = evaluator(pos_score, neg_score)
            ## computation latency
            toc = time.time() - tic

            if pretrain_opts['triple_loss_enable']:
                totalTripleLoss[k] = triple_loss.data[0]
                print "Cycle %2d, K %2d (%2d), BinLoss %.3f, TripleLoss %.3f, Prec %.3f, Time %.3f" % \
                  (i, j, k, cls_loss.data[0], triple_loss.data[0], prec[k], toc)
            else:
                print "Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, k, cls_loss.data[0], prec[k], toc)

        ##########################################validation#################################################
        #####################################################################################################
        if pretrain_opts['validation_enable'] :
            if (i%pretrain_opts['validation_interval'])==0:
                print '================================ validation ================================='
                if pretrain_opts['use_gpu']:
                    tmp_states = {'shared_layers': model.cpu().layers.state_dict()}
                else:
                    tmp_states = {'shared_layers': model.layers.state_dict()}

                torch.save(tmp_states, './models/tmp_model.pth')
                opts['model_path'] = './models/tmp_model.pth'
                if (pretrain_opts['set_type'] == 'VOT') or (pretrain_opts['set_type'] == 'IMAGENET'):
                    opts['set_type'] = 'OTB'
                    seqs = ['Liquor','Toy','Freeman4','Human5','Human3','Matrix','Ironman','Rubik','FaceOcc2','Woman','Singer2','Basketball', 'Skating1', 'MotorRolling', 'Panda', 'Coupon', 'Soccer', 'Bolt', 'Bolt2', 'BlurOwl', 'CarScale','Sylvester','Football']
                    seq_home = '/home/ilchae/dataset/tracking/OTB'
                    seq_list = [seq_home + '/' + p for p in seqs]
                total_iou = []
                opts['visual_log']=False
                opts['short_update_enable']=True
                opts['multi_scale_infer'] = False
                for i in range(0, len(seq_list)):
                    img_list, gt = genConfig(seq_list[i], opts['set_type'])
                    try:
                        iou_result, result_bb, fps = run_mdnet(img_list, gt[0], gt, display=False)
                        print '{} : {} , fps:{}'.format(seq_list[i], iou_result.mean(), fps)
                        total_iou.append(iou_result.mean())
                    except:
                        print '{} fail'.format(seq_list[i])
                try:
                    cur_score = sum(total_iou)/len(seqs)
                except:
                    cur_score = 0.
                if pretrain_opts['use_gpu']:
                    model = model.cuda()
        ##get argument
        else:
            cur_score = prec.mean()
        ##########################################################################################################
        ##########################################################################################################
        cur_triple_loss = totalTripleLoss.mean()
        try:
            total_miou = sum(total_iou)/len(total_iou)
        except:
            total_miou = 0.
        print "Mean Precision: %.3f Triple Loss: %.3f IoU: %.3F" % (prec.mean(), cur_triple_loss, total_miou)
        if cur_score > best_score:
            best_score = cur_score
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print "Save model to %s" % pretrain_opts['model_path']
            torch.save(states, pretrain_opts['model_path'])
            if pretrain_opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'VOT' )
    parser.add_argument("-padding_ratio", default = 5., type =float)
    parser.add_argument("-model_path", default =".models/result_model.pth", help = "model path")
    parser.add_argument("-triple_loss_enable", default = False, action='store_true', help = "enable the appliance of triplet loss")
    parser.add_argument("-validation_enable", default = False, action='store_true',help = "enable the execution of validation experiment")
    parser.add_argument("-frame_interval", default = 1, type=int, help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-validation_interval",default = 10, type=int, help = "We just validate our trained model every \"validation\" iteration")
    parser.add_argument("-init_model_path", default="./models/imagenet-vgg-m.mat")
    parser.add_argument("-batch_frames", default = 16, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)
    parser.add_argument("-efficient_data_prov", default=True, action = 'store_false', help = 'data prov strategy change')
    parser.add_argument("-batch_pos",default = 64, type = int)
    parser.add_argument("-batch_neg", default = 196, type = int)
    parser.add_argument("-n_cycles", default = 1000, type = int )
    parser.add_argument("-adaptive_align", default = False, action = 'store_true')
    parser.add_argument("-seqbatch_size", default=4, type=int)

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio']=args.padding_ratio
    pretrain_opts['padded_img_size']=pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
    pretrain_opts['model_path']=args.model_path
    pretrain_opts['triple_loss_enable'] = args.triple_loss_enable
    pretrain_opts['frame_interval'] = args.frame_interval
    pretrain_opts['validation_enable'] = args.validation_enable
    pretrain_opts['validation_interval']=args.validation_interval
    pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['efficient_data_prov']=args.efficient_data_prov
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align']=args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################

    print pretrain_opts
    train_mdnet()

