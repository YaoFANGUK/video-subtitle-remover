# Current algorithm

## Choice of mask objects

For identification of the objects which are suitable for mask obtaining, panoptic segmentation model
from [detectron2](https://github.com/facebookresearch/detectron2) trained on COCO. Categories of the detected instances
belong either to "stuff" or "things" types. We consider that instances of objects should have category belong
to "things". Besides, we set upper bound on area which is taken by the object &mdash; we consider that too big
area indicates either of the instance being a background or a main object which should not be removed. 

## Choice of position for mask

We consider that input image has size 2^n x 2^m. We downsample it using 
[COUNTLESS](https://github.com/william-silversmith/countless) algorithm so the width is equal to
64 = 2^8 = 2^{downsample_levels}. 

### Augmentation

There are several parameters for augmentation:
- Scaling factor. We limit scaling to the case when a mask after scaling with pivot point in its center fits inside the
 image completely.
-

### Shift 


## Select 
