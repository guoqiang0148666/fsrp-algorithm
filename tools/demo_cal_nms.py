#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from datasets.voc_eval import parse_rec
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from shutil import rmtree

CLASSES = ('__background__',
           'person',
           )

#NETS = {'vgg16': ('VGG16',
#                  'VGG16_faster_rcnn_final.caffemodel'),
#        'zf': ('ZF',
#                  'ZF_faster_rcnn_final.caffemodel')}
#IMGS = {'sy2': ['shenyang1_frame940.jpg', 'shenyang2_frame468.jpg', 'shenyang2_frame128.jpg',
#     'shenyang2_frame126.jpg', 'shenyang1_frame1266.jpg'],
#        'mall': ['seq0.jpg', 'seq1.jpg', 'seq2.jpg', 'seq3.jpg', 'seq4.jpg'],
#        'cuhk': ['set00-occ_0.jpg', 'set00-occ_1.jpg', 'set00-occ_2.jpg', 'set00-occ_3.jpg'
#     ,'set00-occ_1.jpg'] }
IMGS = {'sy2': 'sy-anno2',
        'mall': 'mall',
        'cuhk': '', 
        'cal': 'All-Pedestrian-Dataset',
        'city': 'cityperson/cps_dvkt_w10hv50_ignore' }

CONFS = {'sy-anno2': 0.8,
        'mall': 0.8,
        'cuhk-occ': 0.8, 
        'All-Pedestrian-Dataset': 0.0005,
        'cityperson/cps_dvkt_w10hv50_ignore': 0.8 }

def vis_detections(im, class_name, dets, idx, gtboxes, subdir, thresh=0.5):
    """Draw detected bounding boxes."""
    fout = open('det_scr_box_{}_nms.txt'.format(subdir), 'a')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print >> fout, '{} 0.0 0 0 0 0'.format(idx)
        #print 'image {} no det'.format(idx)
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #print 'bbox {} score {}' .format(bbox, score)

        txt = "Person:%f " %(score)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2, lineType=8, shift=0)
        cv2.putText(im, txt, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1) 
        print >> fout, '{} {} {} {} {} {}'.format(idx, score, bbox[0], bbox[1], bbox[2], bbox[3])

    for i in range(gtboxes.shape[0]):
        gt = np.asarray(gtboxes[i], dtype=np.int)
        cv2.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 255), thickness=2, lineType=8, shift=0)

    gts_txt = "GTs: {}" .format(gtboxes.shape[0])
    dets_txt = "Dets: {}" .format(len(inds))
    
    cv2.putText(im, gts_txt, (480, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1) 
    cv2.putText(im, dets_txt, (480, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1) 
    save = './vis_jpg/' + idx + '.jpg'
    
    cv2.imwrite(save, im)

def demo(net, im_file, dataset, subdir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    recs = {}
    basename = os.path.basename(im_file).split('.')[0]
    xml = '/work/guo-store-files/{}/VOC2007/Annotations/{}.xml' .format(dataset, basename)
    recs[basename] = parse_rec(xml)

    R = [obj for obj in recs[basename] if obj['name'] == 'person']

    gtboxes = np.array([x['bbox'] for x in R])

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, basename)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = CONFS[dataset]
    NMS_THRESH = 0.3

    #f = open('py_confidence.log', 'a')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        #sorted_scores = sorted(cls_scores)
        #print >> f, 'len {}' .format(len(sorted_scores))
        #print >> f, '{}' .format(sorted_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, basename, gtboxes, subdir, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', help='Network to use [vgg16]',
                       default='')
    parser.add_argument('--model', help='weights to use [vgg16]',
                       default='')
    parser.add_argument('--dataset', help='weights to use [vgg16]',
                       default='cal')
    parser.add_argument('--subdir', help='weights to use [vgg16]',
                       default='test')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    # sqznet
    #prototxt = 'sqznet.pt'
    #caffemodel = 'sqznet.caffemodel'
    # mynet
    prototxt = args.net
    caffemodel = args.model
    subdir = args.subdir
    # googlenet
    #prototxt = 'models/pascal_voc/GoogLeNet/faster_rcnn_end2end/test_m5.prototxt'
    #caffemodel = 'output/faster_rcnn_end2end/voc_2007_trainval/crowd_googlenet_m5_iter_100000.caffemodel'

    #prototxt = 'models/pascal_voc/GoogLeNet/faster_rcnn_end2end/test.prototxt'
    #caffemodel = 'output/faster_rcnn_end2end/voc_2007_trainval/googlenet_iter_70000.caffemodel'
    # vgg            
    #prototxt = 'models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    #caffemodel = 'output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_100000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    detfile = 'det_scr_box_{}_nms.txt'.format(subdir)
    if os.path.exists(detfile):
        os.remove(detfile)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #for i in xrange(2):
    #    _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    dataset = IMGS[args.dataset]
    im_names = open('/work/guo-store-files/{}/VOC2007/ImageSets/Main/{}.txt'.format(dataset, subdir)).readlines()
    for im_name in im_names:
        im_path = '/work/guo-store-files/{}/VOC2007/JPEGImages/{}.jpg'.format(dataset, im_name.strip())
        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'Demo for data/demo/{}'.format(im_path)
        demo(net, im_path, dataset, subdir)
