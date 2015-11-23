import os
import sys
import numpy as np
import fileSystemUtils as fs
import cv2 as cv2
import cv2.cv as cv
import lmdb
import scipy.io
import os
import numpy as np
from scipy import io
import lmdb
from read_img import read_img_cv2
NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'
CAFFE_ROOT = '../../'
phase = 'trainval'
sys.path.insert(0, CAFFE_ROOT + 'python/')
print os.listdir(CAFFE_ROOT + 'python')
import caffe
src_dir = CAFFE_ROOT+'data/fcn/'
dst_dir = './'
lmimgDst   = dst_dir + phase +'_imgs_lmdb/'
lmlabelDst = dst_dir + phase + '_labels_lmdb/'


def main(args):
	dir_imgs = CAFFE_ROOT+'data/fcn/' + phase + '_jpg'
	paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)

	dir_segm_labels = CAFFE_ROOT + 'data/fcn/' + phase + '_maps'
	paths_segm_labels = fs.gen_paths(dir_segm_labels)

	paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
	paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
	if not os.path.exists(lmimgDst):
		print 'lmdb dir not exists,make it'
		os.makedirs(lmimgDst)
	if not os.path.exists(lmlabelDst):
		print 'lmdb dir not exists,make it'
		os.makedirs(lmlabelDst)

	size1 = imgs_to_lmdb(paths_imgs, lmimgDst, CAFFE_ROOT = CAFFE_ROOT)
	size2 = matfiles_to_lmdb(paths_segm_labels, lmlabelDst, 'gt_pad',CAFFE_ROOT = CAFFE_ROOT)
	dif = size1 - size2
	dif = dif.sum()
	scipy.io.savemat('./size1',dict({'sz':size1}),appendmat=True)
	scipy.io.savemat('./size2',dict({'sz':size2}),appendmat=True)
	print 'size dif:'+str(dif)
	return 0

def imgs_to_lmdb(paths_src, path_dst, CAFFE_ROOT=None):
    '''
    Generate LMDB file from set of images
    '''
    import numpy as np
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0, CAFFE_ROOT + 'python')
    import caffe
    
    db = lmdb.open(path_dst, map_size=int(1e12))
    size = np.zeros([len(paths_src), 2])
    with db.begin(write=True) as in_txn:
        i = 1
        for idx, path_ in enumerate(paths_src):
            print str(i)+' of '+str(len(paths_src))+' ...'
            
            img = read_img_cv2(path_)
            size[i-1, :] = img.shape[1:]
            img_dat = caffe.io.array_to_datum(img)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
            i = i + 1
    db.close()
    return size

def matfiles_to_lmdb(paths_src, path_dst, fieldname,
                     CAFFE_ROOT=None,
                     lut=None):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
    import caffe
    db = lmdb.open(path_dst, map_size=int(1e12))
    size = np.zeros([len(paths_src), 2])
    with db.begin(write=True) as in_txn:
        i = 1
        for idx, path_ in enumerate(paths_src):
            print str(i)+' of '+str(len(paths_src))+' ...'
            
            content_field = io.loadmat(path_)[fieldname]
            #print content_field.shape
            content_field = np.expand_dims(content_field, axis=0)   ##########
            content_field = content_field.astype(float)
            
            if lut is not None:
                content_field = lut(content_field)
            size[i-1, :] = content_field.shape[1:]
            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
            i = i + 1
    
    db.close()
    return size

if __name__ == '__main__':
     main(None)
     pass
