# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
np.set_printoptions(threshold=np.inf)
import yaml, cv2, time, glob, os
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# sklearn adaboost
#from sklearn.externals import joblib
#import channel_features as cf
# svm
#import UniformLBPDescriptor as ULBP
#from sklearn import svm
#gfrom skimage.feature import hog

# Keras
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.layers import Flatten, Dense, add, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers import BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate, GlobalAveragePooling2D, Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.utils import np_utils
from keras import regularizers, layers
from keras import initializers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MPATH = 'data/imagenet_models/Refined_LeNetbestOK.h5'
DEBUG = False
SAVED = False
PNUMS = 300
WIDTH = 64
HEIGHT = 64
CLASSES = ['background', 'person']

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self.bdt32 = self.LeNet(MPATH)

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(PNUMS, 2)


    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        img = np.squeeze(bottom[3].data)
        channel_swap = (1, 2, 0)  
        img = img.transpose(channel_swap)
        im_orig = img + cfg.PIXEL_MEANS 
        #data = bottom[4].data
        #print img.shape
        #h, w = img.shape[:2]
        #org_h, org_w = (h/im_info[2], w/im_info[2])
        #print org_h, org_w
        #im_orig = cv2.resize(img, (int(org_w), int(org_h)), interpolation=cv2.INTER_CUBIC)
        #print imgname
        #if flipped: print '%s' % (imgname)

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        #print 'proposal shape {} from bbox_transform_inv' .format(proposals.shape[0])

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        #print 'proposal shape {} from filter' .format(proposals.shape[0])
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        #print 'proposal shape {} from topN' .format(proposals.shape[0])

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        #print 'proposal shape {} from nms' .format(proposals.shape[0])

        #out_rois = proposals / im_info[2]
        samples = []
        timer = Timer()
        timer.tic()
        tempname = time.time()
        pnums = proposals.shape[0]
        #print 'timename: ', tempname

        timer.toc()
        #print ('Image processing took {:.3f}s for '
        #     '{:d} object proposals').format(timer.total_time, out_rois.shape[0])

        #print X_in.dtype
        #print >> fout, X_in
        timer = Timer()
        timer.tic()
        remove_keep = []
        if True:
            for idx in range(pnums):
                coords = proposals[idx, :]
                x1, y1, x2, y2 = coords.astype(int)
                im = im_orig[y1:y2, x1:x2].copy()
                im *= 1./255
                re_img = cv2.resize(im, (WIDTH, HEIGHT))
                yp = self.bdt32.predict(re_img[np.newaxis,:])[0,1]
                scr = scores[idx]
                #cls = CLASSES[int(y)]
                #outpath = 'temp/{}/{}_clip{:03d}.jpg'.format(cls, tempname, idx)
                if yp <= 0.4 and scr <= 0.9:
                #if scr <= 0.9:
                    remove_keep.append(idx)
                if False:
                    #im_orig = im + cfg.PIXEL_MEANS 
                    #y = self.bdt32.predict_classes(re_img[np.newaxis,:])
                        #print idx, yp, scr 
                        outpath = 'temp/ba/{}_clip{:03d}.jpg'.format(tempname, idx)
                        #outpath_org = 'ada/proposals/personorg_clip{:03d}.jpg'.format(idx)
                        save_img = re_img.copy()
                        save_img *= 255
                        cv2.imwrite(outpath, save_img)
            #if img.shape[0] <= 4 or img.shape[1] <=4: continue
            #gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)
            #break
            #samples.append(re_img)
            #X_in = np.array(samples)
            #y_prob = self.bdt32.predict(X_in)
            #remove_keep = np.where(y_prob[:, 1]<=0.1)[0]
            #check_scrs = scores[remove_keep][0]
            #print check_scrs, remove_keep
            #remove_keep_update = remove_keep[scores[remove_keep]<0.5]
            ada_keep = np.setdiff1d(np.arange(pnums), remove_keep)
            if ada_keep.size != 0:
                proposals = proposals[ada_keep, :]
            else:
                proposals = proposals[0, :]
                proposals = proposals[np.newaxis, :]
        #y = self.bdt32.predict_classes(X_in)
        #y = y_prob.argmax(axis=-1)
        #print y.shape
        #print y
        #y_prob = self.bdt32.decision_function(X_in)
        timer.toc()
        #print ('Ada detection took {:.3f}s for '
        #     '{:d} object proposals').format(timer.total_time, proposals.shape[0])
        #ada_keep = np.where(y==1)[0]
        #ada_keep = np.where(y_prob[:, 1]>=0.8)[0]
        #ada_keep = np.where(scores>=0.5)[0]
        #print ada_keep
        #print y_prob[ada_keep]
        #assert False, 'Stop'
        #ada_keep = np.where(y_prob>0.5)[0]
        #print y_prob.shape, X_in.shape, ada_keep.shape

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        #print proposals
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(y_prob.shape))
            top[1].data[...] = y_prob


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def LeNet(self, weights_path=None, heatmap=False):
    
        input_shape=(64,64,3)
        
        #seed = 7  
        #np.random.seed(seed)  
      
        model = Sequential()  
        model.add(Conv2D(20,(3,3),strides=(1,1),input_shape=input_shape,padding='valid',activation='relu', \
                  kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Constant()))  
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  
        model.add(Conv2D(40,(3,3),strides=(1,1),input_shape=input_shape,padding='valid',activation='relu', \
                  kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Constant()))  
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  
        model.add(Conv2D(60,(3,3),strides=(1,1),input_shape=input_shape,padding='valid',activation='relu', \
                  kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Constant()))  
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  
        model.add(Conv2D(80,(3,3),strides=(1,1),input_shape=input_shape,padding='valid',activation='relu', \
                  kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Constant()))  
    
        model.add(Flatten())
        model.add(Dense(500, activation='relu', kernel_initializer=initializers.glorot_normal(),\
                  bias_initializer=initializers.Constant()))  
        model.add(Dense(2, activation='relu', kernel_initializer=initializers.glorot_normal(),\
                  bias_initializer=initializers.Constant()))  
        model.add(Activation("softmax", name="softmax"))
    
    
        if weights_path and os.path.exists(weights_path):
            print('model from %s' %weights_path)
            model = load_model(weights_path)
    
        return model

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
