# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
from datasets.voc_eval import parse_rec
import cPickle
from utils.blob import im_list_to_blob
from utils.cython_bbox import bbox_overlaps
import os

#upper = cfg.TEST.RPN_POST_NMS_TOP_N

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v).astype(np.int)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        upper = net.blobs['ps'].data
        # unscale back to raw image space
        #boxes = rois[:upper, 1:5] / im_scales[0]
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        #box_deltas = blobs_out['bbox_pred'][:upper, :]
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, int(np.squeeze(upper))

def vis_detections(im, class_name, gtboxes, idx, dets, thresh=0.3):
    """Visual debugging of detections."""
    '''
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()
    '''
    inds = dets.shape[0]
    if inds == 0: return

    for i in range(inds):
        bbox = dets[i, :4]
        score = dets[i, -1]

        #print 'bbox {} score {}' .format(bbox, score)

        txt = "Person: %f " %(score)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2, lineType=8, shift=0)
        if bbox[1] <= 20:
            cv2.putText(im, txt, (bbox[0], int(bbox[1]+20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1) 
        else:
            cv2.putText(im, txt, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1) 

    for i in range(gtboxes.shape[0]):
        gt = np.asarray(gtboxes[i], dtype=np.int)
        cv2.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 255), thickness=2, lineType=8, shift=0)

    if False:
        cv2.imshow('test', im)
        if (cv2.waitKey(300) & 0xff) == 27: return

    cv2.imwrite('vis_jpg/{}.jpg'.format(idx), im)

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.0005, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    CROP_THRESH = 0.01
    #CROP_THRESH = 0.001

    rootdir = '{}/data/VOCdevkit2007/VOC2007/'.format(os.getcwd())
    filenames = open(os.path.join(rootdir, 'ImageSets/Main/test.txt')).readlines()


    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    tot = tot_nf = fps = ffps = 0

    for i in xrange(num_images):
        # filter out any ground truth boxes
        recs = {}
        basename = filenames[i].strip()
        print basename
        xml = '{}/Annotations/{}.xml' .format(rootdir, basename)
        recs[basename] = parse_rec(xml)
        R = [obj for obj in recs[basename] if obj['name'] == 'person']
        gtboxes = np.array([x['bbox'] for x in R])

        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, upper = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:upper, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            cls_boxes_all = boxes[:, j*4:(j+1)*4]
            cls_scores_all = scores[:, j]
            dets_all = np.hstack((cls_boxes_all,
                              cls_scores_all[:, np.newaxis])).astype(np.float32)
            #keep_nms = nms(cls_dets, cfg.TEST.NMS)

            keep = nms(cls_dets, cfg.TEST.NMS)

            kmap = np.array(inds[keep])
            nlim = kmap.size
            #keep_all = kmap.copy()
            rm_idx = []
            #for it in range(upper, upper+2): keep_all = np.append(keep_all, it+kmap*2)
            for idx, ii in enumerate(kmap):
                minscr = min(cls_scores_all[ii*2 + upper], cls_scores_all[ii*2 + upper + 1])
                x1, y1, x2, y2 = cls_dets[keep[idx], :4]
                w = x2 - x1
                h = y2 - y1
                amax = max(w,h)
                amin = min(w,h)
                #if amax > 200 and amin > 100: 
                #if w > h: 
                #    rm_idx.append(idx)
                #    print basename, w, h
                #    continue
                #if h > 100 and w < 40:
                #    rm_idx.append(idx)
                #    print basename, w, h
                #    continue
                #print minscr, cls_scores_all[ii], cls_dets[keep[idx], -1]
                #if minscr < CROP_THRESH:
                #if cls_scores_all[ii*2 + upper]<= CROP_THRESH or cls_scores_all[ii*2 + upper + 1] <= CROP_THRESH:
                #if minscr < CROP_THRESH and amin >=100 and cls_dets[keep[idx], -1] <= 0.9:
                #if cls_dets[keep[idx], -1] < 0.001 and cls_scores_all[ii*2 + upper]<= CROP_THRESH and cls_scores_all[ii*2 + upper + 1] <= CROP_THRESH:
                if cls_scores_all[ii*2 + upper]<= CROP_THRESH and cls_scores_all[ii*2 + upper + 1] <= CROP_THRESH:
                #if minscr < CROP_THRESH and amin >=100:
                    #print minscr, cls_scores_all[ii], cls_dets[keep[idx], -1]
                    #if w >= h or max(w, h) >= 200: 
                    rm_idx.append(idx)
                    print basename, w, h, cls_dets[keep[idx], -1], cls_scores_all[ii*2 + upper], cls_scores_all[ii*2 + upper + 1]
                        #print ii, minscr, w, h, cls_scores_all[ii], cls_dets[keep[idx], -1]
                    #print minscr, w, h
                #    rm_idx.append(idx)
            ##    le_rt_scr = (scores[i*2 + 100] + cls_scores_all[i*2 + 101]) / 2.0 
            ##    if le_rt_scr <= CROP_THRESH: le_rt_remove.append(i)
                #if cls_scores_all[ii*2 + upper]<= CROP_THRESH and cls_scores_all[ii*2 + upper + 1] <= CROP_THRESH:
                #    rm_idx.append(idx)
                #print minscr, cls_scores_all[ii], cls_dets[keep[idx], -1]
                #if cls_scores_all[ii] < 0.2 and minscr <= CROP_THRESH:
                #print ii, cls_scores_all[ii*2 + upper], cls_scores_all[ii*2 + upper + 1]
            le_rt_remove = [ keep[int(iii)] for iii in rm_idx ]
            false_fps = []
            #for jj in le_rt_remove:
            #    bb = cls_dets[jj, :4]
            #    overlaps = bbox_overlaps(
            #        np.ascontiguousarray(bb[np.newaxis, :], dtype=np.float),
            #        np.ascontiguousarray(gtboxes, dtype=np.float))
            #    max_overlap = overlaps.max(axis=1)
            #    if max_overlap >= 0.1: false_fps.append(jj)
            #le_rt_remove = []
            #false_fps = []
            #for jj in keep:
            #    bb = cls_dets[jj, :4]
            #    overlaps = bbox_overlaps(
            #        np.ascontiguousarray(bb[np.newaxis, :], dtype=np.float),
            #        np.ascontiguousarray(gtboxes, dtype=np.float))
            #    max_overlap = overlaps.max(axis=1)
            #    if max_overlap <= 0.1: 
            #        print basename, bb[2] - bb[0], bb[3] - bb[1]
            #        #print cls_dets[jj, -1], cls_scores_all[inds[jj]], cls_scores_all[inds[jj]*2 + upper], cls_scores_all[inds[jj]*2 + upper + 1]
            #        le_rt_remove.append(jj)
            #print le_rt_remove
            #print keep
            #le_rt_remove = np.setdiff1d(le_rt_remove, false_fps)
            tot += len(keep)
            fps += len(le_rt_remove)
            ffps += len(false_fps)

            kkmap = np.array(inds[le_rt_remove])
            nnlim = kkmap.size
            keep_all = kkmap.copy()
            for it in range(upper, upper+2): keep_all = np.append(keep_all, it+kkmap*2)

            keep = np.setdiff1d(keep, le_rt_remove)
            tot_nf += len(keep)
            if len(keep) != 0:
                cls_dets = cls_dets[keep, :]
            else:
                cls_dets = []
            #print keep, le_rt_remove
            #print cls_dets.shape[0]
            #print cls_dets[:,-1]
            if vis:
                #vis_detections(im, imdb.classes[j], dets_all[keep_all[nlim:], :])
                #vis_detections(im, imdb.classes[j], gtboxes, dets_all[kmap[rm_idx], :])
                #vis_detections(im, imdb.classes[j], gtboxes, basename, cls_dets)
                vis_detections(im, imdb.classes[j], gtboxes, basename, dets_all[keep_all[nnlim:], :])
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        #if max_per_image > 0:
        #    image_scores = np.hstack([all_boxes[j][i][:, -1]
        #                              for j in xrange(1, imdb.num_classes)])
        #    if len(image_scores) > max_per_image:
        #        image_thresh = np.sort(image_scores)[-max_per_image]
        #        for j in xrange(1, imdb.num_classes):
        #            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
        #            all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        #print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        #      .format(i + 1, num_images, _t['im_detect'].average_time,
        #              _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'total dets {} tot_nf {} fps {} ffps {}'.format(tot, tot_nf, fps, ffps)
    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
