import os,sys
import argparse
import numpy as np
from glob import glob
from shutil import rmtree
import cPickle
import cv2
import matplotlib.pyplot as plt

import _init_paths
from datasets.voc_eval import voc_eval

parser = argparse.ArgumentParser(description='Caltech eval')

parser.add_argument('-dpath', dest='devpath',default='',required=True,
                   help='devpath filename', type=str)
parser.add_argument('-rpath', dest='respath',default='',required=True,
                   help='devpath filename', type=str)

parser.set_defaults(run_soon=True)

args = parser.parse_args()

devkit_path = args.devpath
result_path = args.respath

COLOR = {'FPN':'dodgerblue',
         'Tiny MetroNext':'darkorchid',
         'Tiny YOLOV3':'crimson',
         'MetroNext':'orange' }

#ls = ['-', '-.', ':', '--']
ls = {'MetroNext':'-',
      'Tiny MetroNext':':',
      'Tiny YOLOV3':'--',
      'FPN':'-.'}

classes = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

classes = ('__background__', # always index 0
            'person')

def _do_python_eval(output_dir = 'output'):
    annopath = os.path.join(
        devkit_path,
        'VOC2007',
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        'test.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for mod in ['FPN', 'MetroNext', 'Tiny MetroNext', 'Tiny YOLOV3']:
        #ax = plt.subplot(111)
        for i, cls in enumerate(classes):
            if cls == '__background__':
                continue
            filename = '{}_cal_det_scr_box.txt'.format(mod.lower().replace(' ', ''))
            filepath = os.path.join(result_path, \
                       filename)
            rec, prec, ap = voc_eval(
                filepath, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            	
            plt.plot(rec, prec, lw=1, color=COLOR[mod], linestyle=ls[mod],
                  label='{}'
                        ''.format(mod))

            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='14')
        plt.xlabel('Recall', fontsize='12')
        plt.ylabel('Precision', fontsize='12')
	plt.grid(True, linestyle = "-.")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall', fontsize='12')
        legend = plt.legend(loc="lower right", title='Caltech Reasonable', fontsize='12') 
        legend.get_title().set_fontsize('12')
        #plt.savefig('test.eps', format='eps', bbox_inches='tight', dpi=1000)
        plt.savefig('test.pdf', bbox_inches='tight', dpi=1000)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

if __name__ == '__main__':
    _do_python_eval()
