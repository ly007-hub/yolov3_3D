from lyi.utils import *


nii_path = r'E:\ly\pnens_data\nas_data\v1_data\NIH\data\PANCREAS_0030.nii.gz'
mask_path = r'E:\ly\pnens_data\nas_data\v1_data\NIH\mask\label0030.nii.gz'
img = read_nii(nii_path)
mask = read_nii(mask_path)
show3D(img)
show3D(mask)
show2D(img[:, :, 120])
for i in range(mask.shape[2]):
    show2D(mask[:, :, i])
"""
冠状面、矢状面、水平面[y, x, z]
img [128, 128, 64]
bbox[y1, y2, x1, x2, z1, z2]
mask[128, 128, 64]
"""
dataset2 = load_from_pkl(r'E:\ly\pnens_data\nas_data\v1_data\NIH\pre_order0_128_128_64_new.pkl')
data_0 = dataset2[0]['train_set'][0]
bbox = data_0['bbox']
img = data_0['img']
mask = data_0['mask']
img_mask = img + mask * 1000
show3Dslice(img_mask)
y1, y2, x1, x2, z1, z2 = bbox
bbox_1 = [y1, x1, z1, y2, x2, z2]
data_0_img_bbox = bbox_in_img(img, np.array(bbox_1).astype(int), 2)
show3Dslice(data_0_img_bbox)

data_0['img'].shape
data_0['img'].max()
data_0['img'].min()
data_0['mask'].shape
data_0['mask'].max()
data_0['mask'].min()
data_1 = dataset2[0]['train_set'][1]
data_1['img'].shape
data_1['img'].max()
data_1['img'].min()

show3D(data_0['img'])
show2D(data_0['img'][:, :, 32])
show3D(data_0['mask'])
show2D(data_0['mask'][:, :, 32])
img_mask_0 = data_0['mask']*1e4 + data_0['img']
show3D(img_mask_0)
for i in range(img_mask_0.shape[2]):
    show2D_z(img_mask_0, i)
show2D_z(img_mask_0, 64)
