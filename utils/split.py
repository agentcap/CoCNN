import os
import numpy as np
from shutil import copyfile

img_lst = os.listdir('all_images')
np.random.shuffle(img_lst)

trn_sz = int(len(img_lst) * 0.75)
tst_sz = len(img_lst) - trn_sz

trn_lst = img_lst[:trn_sz]
tst_lst = img_lst[trn_sz:]


#print("No of training images ", trn_lst)
#print('-'*40)
train_dir = 'trainset/'
if not os.path.exists(train_dir+'JPEGImages/'):
    os.makedirs(train_dir+'JPEGImages/')
if not os.path.exists(train_dir+'SegmentationClassAug/'):
    os.makedirs(train_dir+'SegmentationClassAug/')
for nm in trn_lst:
	copyfile('all_images/'+nm, train_dir+'JPEGImages/'+nm)
	print('all_images/'+nm, train_dir+'JPEGImages/'+nm)
	lb_nm = nm.split('.')[0] + '.png'
	copyfile('all_images_output/'+lb_nm, train_dir+'SegmentationClassAug/'+lb_nm)
	print('all_images_output/'+lb_nm, train_dir+'SegmentationClassAug/'+lb_nm)
	print('-'*10)

test_dir = 'testset/'
if not os.path.exists(test_dir+'JPEGImages/'):
    os.makedirs(test_dir+'JPEGImages/')
if not os.path.exists(test_dir+'SegmentationClassAug/'):
    os.makedirs(test_dir+'SegmentationClassAug/')
for nm in tst_lst:
	copyfile('all_images/'+nm, test_dir+'JPEGImages/'+nm)
	lb_nm = nm.split('.')[0] + '.png'
	copyfile('all_images_output/'+lb_nm, test_dir+'SegmentationClassAug/'+lb_nm)
	 
