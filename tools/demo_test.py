import _init_paths
import numpy as np
import cv2,sys
import caffe

#from rknn.api import RKNN

_, MODEL, WEIGHT, IMGPATH = sys.argv

INPUT_SIZE_LONG = 1000
INPUT_SIZE_NARROW = 600
BBOX_XFORM_CLIP = np.log(1000. / 16.) 

CONF_THRESH = 0.8
NMS_THRESH = 0.2

CLASSES= ('__background__', # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__',
           'person'
           )

def vis_detections(im_orig, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        print('bbox {} score {}'.format(bbox, score))

        txt = "%s: %f" %(class_name, score)
        cv2.rectangle(im_orig, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2, lineType=8, shift=0)
        cv2.putText(im_orig, txt, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1) 

    dets_txt = "Dets: {}" .format(len(inds))
    cv2.putText(im_orig, dets_txt, (480, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1) 


def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline."""              
    x1 = dets[:, 0]                    
    y1 = dets[:, 1]                                          
    x2 = dets[:, 2] 
    y2 = dets[:, 3]       
    scores = dets[:, 4] 
                                         
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)                 
    order = scores.argsort()[::-1]   
                                    
    keep = []                                                                   
    while order.size > 0:                                                       
        i = order[0]                                                            
        keep.append(i)                                                          
        xx1 = np.maximum(x1[i], x1[order[1:]])                                  
        yy1 = np.maximum(y1[i], y1[order[1:]])                                  
        xx2 = np.minimum(x2[i], x2[order[1:]])                                  
        yy2 = np.minimum(y2[i], y2[order[1:]])                                  
                                                                                
        w = np.maximum(0.0, xx2 - xx1 + 1)                                      
        h = np.maximum(0.0, yy2 - yy1 + 1)                                      
        inter = w * h                                                           
        ovr = inter / (areas[i] + areas[order[1:]] - inter)                     
                                                                                
        inds = np.where(ovr <= thresh)[0]                                       
        order = order[inds + 1]                                                 
                                                                                
    return keep 

def bbox_transform_inv(boxes, deltas):                                          
    if boxes.shape[0] == 0:                                                     
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)               
                                                                                
    boxes = boxes.astype(deltas.dtype, copy=False)                              
                                                                                
    #widths = boxes[:, 2] - boxes[:, 0] + 1.0                                   
    #heights = boxes[:, 3] - boxes[:, 1] + 1.0                                  
    widths = boxes[:, 2] - boxes[:, 0]                                          
    heights = boxes[:, 3] - boxes[:, 1]                                         
    ctr_x = boxes[:, 0] + 0.5 * widths                                          
    ctr_y = boxes[:, 1] + 0.5 * heights                                         
                                                                                
    #print 'anchor w {} h {}'.format(widths, heights)                           
    dx = deltas[:, 0::4]                                                        
    dy = deltas[:, 1::4]                                                        
    dw = deltas[:, 2::4]                                                        
    dh = deltas[:, 3::4]                                                        
                                                                                
    # prevent sending too large values into np.exp()                            
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u

if __name__ == '__main__':

    #prototxt = 'paper_result/voc2007/sqznet/sqznet_voc.pt'
    #caffemodel = 'paper_result/voc2007/sqznet/sqznet_voc_iter_190000.caffemodel'

    #prototxt = 'paper_result/station_weights/rknn-models/metronext.pt'
    #caffemodel = 'paper_result/station_weights/rknn-models/metronext.caffemodel'

    prototxt = MODEL
    caffemodel = WEIGHT

    net = caffe.Net(MODEL, WEIGHT, caffe.TEST)
    # Set inputs
    im_orig = cv2.imread(IMGPATH)
    #im_orig = cv2.imread('./syst2019527-1_0_frame19.jpg')
    img = im_orig.astype(np.float32, copy=True)
    img -= np.array([[[102.9801, 115.9465, 122.7717]]])
    
    im_shape = img.shape 
    #print 'org im shape ', im_shape 
    im_size_min = np.min(im_shape[0:2]) 
    im_size_max = np.max(im_shape[0:2])
    max_size_scale = float(im_size_max) / INPUT_SIZE_LONG
    min_size_scale = float(im_size_min) / INPUT_SIZE_NARROW
    max_scale = max(max_size_scale, min_size_scale)
    img_scale = 1.0 / max_scale
    resize_h = int(im_shape[0] * img_scale)
    resize_w = int(im_shape[1] * img_scale)

    print(resize_h, resize_w, img_scale)

    #img = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
    img = img[np.newaxis, :, :, :]
    channel_swap = (0, 3, 1, 2)
    img = img.transpose(channel_swap)

    print(img.shape)

    net.blobs['data'].reshape(*(img.shape))
    im_info = np.array([[resize_h, resize_w, img_scale]], dtype=np.float32)
    net.blobs['im_info'].reshape(*(im_info.shape))
    forward_kwargs = {'data': img.astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = im_info.astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / img_scale

    preds = blobs_out['bbox_pred']
    pred_scores = blobs_out['cls_prob']
    #pred_refinedscores = blobs_out['subnet_cls_prob']

    print(preds.shape, pred_scores.shape, boxes.shape)

    pred_boxes = bbox_transform_inv(boxes, preds)
    pred_boxes = clip_boxes(pred_boxes, im_shape) 

    for cls_ind, cls in enumerate(CLASSES[1:]):                                
        cls_ind += 1 # because we skipped background                           
        cls_boxes = pred_boxes[:, 4*cls_ind:4*(cls_ind + 1)]                        
        cls_scores = pred_scores[:, cls_ind]
        #sorted_scores = sorted(cls_scores)                                    
        #print >> f, 'len {}' .format(len(sorted_scores))                      
        #print >> f, '{}' .format(sorted_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)       
        keep = py_cpu_nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im_orig, cls, dets, CONF_THRESH)
 
    save = 'results.jpg'
    
    cv2.imwrite(save, im_orig)

    # Evaluate Perf on Simulator
    #rknn.eval_perf(inputs=[img], is_print=True)

    # Release RKNN Context
