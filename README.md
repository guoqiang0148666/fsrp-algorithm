This repository contains code for the paper "Enhancing Pedestrian Detection Accuracy: A Full-Stage Refined Proposal Algorithm for False Positive Suppression" is submmited in [Visual Computer] (https://link.springer.com/journal/371).

### Disclaimer
It is an official implementation built upon the py-faster-rcnn codebase by Ross Girshick.

Disclaimer and Copyright
This project is based on the [py-faster-rcnn] (https://github.com/rbgirshick/py-faster-rcnn.git) developed by Ross Girshick and other contributors. The original code is released under the MIT License, as stated in its repository.

This derivative work is also distributed under the MIT License. The copyright and licensing terms of the original project remain in effect. Please see the LICENSE file in the original repository and this one for more details.

We acknowledge and are grateful for the significant contribution of the original authors.

### Installation
The installation process for this codebase is identical to the original py-faster-rcnn requirements. Please follow the installation instructions provided in the official py-faster-rcnn [README] (https://github.com/rbgirshick/py-faster-rcnn.git/README.md).

### Usage
After successful installation, you can use this code to reproduce experiment results in this paper.

  **NOTE** link the ./data/VOCdevkit2007 to the CityPersons dataset
  ``` run
  python2 ./tools/test_net.py --def paper_result/test.prototxt --net paper_result/model.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
  ```
  The detection results is stored in det\_scr\_box.txt, which can be used to reproduce MR results in CityPersons datasets
Thanks original Faster R-CNN Paper:

bibtex
@inproceedings{ren2015faster,
  title={Faster R-CNN: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  year={2015}
}

Acknowledgements
This code is a modification of the excellent work from:

Ross Girshick (rbg) - py-faster-rcnn

The core Faster R-CNN authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
