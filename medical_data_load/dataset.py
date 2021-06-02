from torchvision.datasets.mnist import read_image_file, read_label_file
# from utils import tensor2numpy, show2D, get_filelist_frompath, list_unique, readimg, list_slide
import os
import csv
import PIL
import pickle
import numpy as np
sep = os.sep
from medical_data_load.utils import *
from medical_data_load.config_dataset import config as data_config
# from wama.utils import *
import SimpleITK as sitk
import medical_data_load.config_dataset
# 医学数据集-3D
# 胰腺（具体名字待补充，CT 的），
miccai_2018_decathlon_data_root = data_config['miccai_2018_decathlon_data_root']
NIH_pancreas_data_root = data_config['NIH_pancreas_data_root']


def read_nii2array4miccai_pancreas(img_pth, mask_pth, aim_spacing, aim_shape, order=3, is_NIH = False, cut_bg=True):
    # img_pth, mask_pth, aim_spacing, aim_shape, order, is_NIH, cut_bg = case['img_path'], case['mask_path'], aim_spacing, aim_shape, order, True, cut_bg
    """
    img_pth, mask_pth, aim_spacing, aim_shape, order, is_NIH, cut_bg = case['img_path'], case['mask_path'], aim_spacing, aim_shape, order, True, cut_bg
    :param img_pth:
    :param mask_pth:
    :param aim_spacing:
    :param aim_shape:
    :param order:
    :param is_NIH: 如果是NIH数据集，则需要调整各个维度顺序，使之和MICCAI一样
    :return:
    """
    # img_pth = case['img_path']
    # mask_pth = case['mask_path']
    # aim_shape = [128,128,64]
    # aim_spacing = [0.5,0.5,0.8]


    image_reader = wama()  # 构建实例
    image_reader.appendImageAndSementicMaskFromNifti('CT', img_pth, mask_pth)

    # # 修正label(原始数据是错的，一定要先修正，如果使用的是经过修正的，就算了）
    # if is_NIH:
    #     image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][::-1,:,:]


    # (不要在这里调整窗宽窗位，因为可能用到多窗宽窗位）
    # image_reader.adjst_Window('CT', 321, 123)
    # resample
    if aim_spacing is not None:
        print('resampling to ', aim_spacing, 'mm')
        image_reader.resample('CT', aim_spacing, order=order)  # 首先resample没得跑,[0.5,0.5,0.8]就好

    # 去除多余部分
    # scan = image_reader.scan['CT']
    if cut_bg:
        print('cuting bg')
        bbox = remove_bg4pancreasNII(image_reader.scan['CT'])
        image_reader.scan['CT'] = image_reader.scan['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]


    # resize到固定大小
    scan = resize4pancreasNII(image_reader.scan['CT'], aimspace=aim_shape, order=order, is_mask=False, is_NIH= is_NIH)
    mask = resize4pancreasNII(image_reader.sementic_mask['CT'], aimspace=aim_shape, order=0, is_mask=True, is_NIH=is_NIH)  # 注意mask是order 0
    shape = np.array(image_reader.sementic_mask['CT'].shape) / np.array(mask.shape)
    shape = [shape[0], shape[0], shape[1], shape[1], shape[2], shape[2]]
    # resize bbox
    bbox =  np.array(image_reader.bbox['CT']) / np.array(shape)

    # 由于mask存在肿瘤和胰腺，而我们只需保存胰腺即可，所以要把胰腺和肿瘤合并！
    mask = (mask >= 0.5).astype(mask.dtype)

    return scan, bbox


def resize4pancreasNII(scan, aimspace = [64, 64, 64], order = 0, is_mask = False, is_NIH = False):
    """
    # 由于医学图像的特性，xy分辨率较高，所以首先对image x和y进行resize，z等比例保持不变
    # 1）如果z稍微大于这目标z，则resize到目标z ； 如果z大于z太多，则需要把上面的减裁掉（因为胰腺在下面，所以把肺部减去），减到符合标准再resize
    # 2）如果z略微小于目标，则同样resize到目标z； 否则在上面补0，补到符合标准，再resize
    # aim_size 这个要根据resample后的总体尺寸来定，不要瞎搞，[128,128,64]就差不多（对于这个数据集）

    """
    # scan = image_reader.scan['CT']
    # 保持z轴相对比例,先将xy缩放到对应shape
    scan = resize3D(scan, aimspace[:2]+[(scan.shape[-1]/scan.shape[0])*aimspace[0]], order)


    # 注意，这里以miccai2018 的为标准，调整nii的数据使之方向和miccai一样
    # 也就是需要下面这个操作
    # show3Dslice(scan[:,:,22:])
    # show3Dslice(scan)
    # show3Dslice(scan_NIH[::-1,::-1,::-1])
    # show3Dslice(scan_miccai)
    # if is_NIH:
    #     scan = scan[::-1, ::-1, ::-1]


    if True:
        thresold = (5/64)*aimspace[-1] # todo 自己设定的阈值，64基础上，上下可以差4
        if abs(scan.shape[-1] - aimspace[-1]) <= thresold:
            # 如果z很接近，就直接resize z轴到目标尺寸
            scan = resize3D(scan, aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1])>=0:
            # 如果层数过多，则删除底部（胯部）到阈值+aimspace(因为胰腺一般靠近肝脏和肺部，而不靠近跨部），之后再resize
            # cut_slices = scan.shape[-1] -
            # scan = scan[:,:,:int((aimspace[-1]+thresold))]  # 注意这个顺序 todo 有点问题 mmp，部分label会被切掉，暂时不要这个操作
            scan = resize3D(scan, aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1])<0:
            # 如果层数过多，则在顶部（肺部）添加0层到阈值-aimspace，之后再resize
            cut_slices = abs(scan.shape[-1] - (aimspace[-1]-thresold))
            tmp_scan = np.zeros(aimspace[:2]+[int(scan.shape[-1]+cut_slices)], dtype=scan.dtype)
            if is_mask:
                pass # 如果是分割mask，则赋予0
            else:
                tmp_scan = tmp_scan - 1024 # 如果是CT，则赋予空气值
            tmp_scan[:, :, :scan.shape[-1]] = scan
            scan = resize3D(scan, aimspace, order)
        else:
            scan = resize3D(scan, aimspace, order)

    return scan


def remove_bg4pancreasNII(scan):
    """
    # 此外，CT图像预处理，可以包含“去除非身体的扫描床部分”
    # 也就是去除无关地方，这可以极大减少冗余的地方
    # 这个正好也可以在袁总的数据上用到
    # 一般来说，CT值小于-850 的地方，就可以不要了，不过还是要留一个参数控制阈值
    # 思路：二值化，开操作，取最大联通，外扩，剩下的都不要，完事，记得也要把截取矩阵输出，以供分割使用

    :param scan: 没有卡过窗宽窗外的图像！！！！
    :return:
    """


    # scan = image_reader.scan['CT']
    # scan = resize3D(scan,[256,256,None])
    # show3D(scan)
    # show3D(scan_mask_af)

    scan_mask = (scan> -900).astype(np.int)
    sitk_img = sitk.GetImageFromArray(scan_mask)
    sitk_img = sitk.BinaryMorphologicalOpening(sitk_img!=0, 15)
    scan_mask_af = sitk.GetArrayFromImage(sitk_img)
    # show3Dslice(np.concatenate([scan_mask_af, scan_mask],axis=1))
    scan_mask_af = connected_domain_3D(scan_mask_af)
    # 计算得到bbox，形式为[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    indexx = np.where(scan_mask_af > 0.)
    dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                            np.min(indexx[1]), np.max(indexx[1]),
                                                            np.min(indexx[2]), np.max(indexx[2])]
    return [dim0min, dim0max, dim1min, dim1max]


def get_dataset_NIH_pancreas(preload_cache = False, order = 3):
    print('getting NIH_pancreas')
    dataset_name = r'NIH_pancreas'
    root = data_config['NIH_pancreas_data_root']
    aim_spacing = data_config['NIH_pancreas_data_aimspace']
    aim_shape = data_config['NIH_pancreas_data_aimshape']
    cut_bg = data_config['NIH_pancreas_data_cut_bg']

    nii_list = get_filelist_frompath(root+sep+r'data', 'nii.gz')

    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for case in nii_list:
            train_list.append(dict(
                img=None,
                bbox=None,
                img_path=case,
                mask_path=root+sep+'mask'+sep+'label'+case.split(sep)[-1][-11:],
                ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        """
        for index, case in enumerate(train_list):
            if index == 0:
                break
        """
        for index, case in enumerate(train_list):
            print('loading ', index+1, '/', len(train_list), '...')
            scan, bbox = read_nii2array4miccai_pancreas(case['img_path'], case['mask_path'], aim_spacing, aim_shape, order, True, cut_bg)
            print(scan.shape)
            case['img'] = scan.astype(np.float32)
            case['bbox'] = bbox.astype(np.float32)


    return dict(train_set = train_list,
                 train_set_num = len(train_list),
                dataset_name = dataset_name)


if __name__ == '__main__':
    # dataset= get_dataset_MNIST()
    # dataset= get_dataset_miccai2018pancreas(preload_cache=True, order=0)
    # save_as_pkl(r'/media/szu/2.0TB_2/wmy/@database/miccai_2018_decathlon/Pancreas Tumour_precessed/pre_order0_128_128_64_new.pkl',
    #             [dataset, data_config])

    dataset2 = get_dataset_NIH_pancreas(preload_cache=True, order=0)
    save_as_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_64_64_64_new.pkl',
                [dataset2, data_config])

    # 可以直接变成2D的数据集
    # dataset_test2D = make_dataset_2D(dataset_test, False)
    # show2D(dataset_test2D['train_set'][6]['img'])



    # 检查mask和原图是否匹配
    # show3Dslice(np.concatenate([mat2gray(dataset['train_set'][0]['img']),0.5*mat2gray(dataset['train_set'][0]['img'])+0.5*dataset['train_set'][0]['mask']],axis=1))
    # show3Dslice(np.concatenate([mat2gray(dataset['train_set'][0]['img']),dataset['train_set'][0]['mask']],axis=1))
    # show2D(mat2gray(dataset['train_set'][0]['img'][:,:,32]))
    # show2D(mat2gray(dataset['train_set'][0]['mask'][:,:,32]))
	#
    # show3Dslice(np.concatenate([mat2gray(dataset2['train_set'][0]['img']),0.5*mat2gray(dataset2['train_set'][0]['img'])+0.5*dataset2['train_set'][0]['mask']],axis=1))
    # show3Dslice(np.concatenate([mat2gray(dataset2['train_set'][0]['img']), dataset2['train_set'][0]['mask']], axis=1))
    # show2D(mat2gray(dataset2['train_set'][0]['img'][:, :, 32]))
    # show2D(mat2gray(dataset2['train_set'][0]['mask'][:, :, 32]))
	#
    # # 检查mask是不是二值，以及方向是不是一致
    # show3D((dataset['train_set'][2]['mask']))
    # show3D((dataset2['train_set'][0]['mask']))
	#
    # # 检查原图是不是方向一致
    # # 要求，mayavi显示后，肝脏在右，肺部在上
    # show3D((dataset['train_set'][2]['img']))
    # show3Dslice((dataset2['train_set'][2]['img']))


    # save_as_pkl(r'F:\9database\database\miccai_2018_decathlon\Pancreas Tumour_precessed\pre_order3_128_128_64.pkl',[dataset,data_config] )
    # save_as_pkl(r'F:\9database\database\NIH-pancreas\Pancreas-CT\pre_order3_128_128_64.pkl',[dataset2,data_config] )
	#
	#
	#
    # dataset_pre = load_from_pkl(r'F:\9database\database\miccai_2018_decathlon\Pancreas Tumour_precessed\pre_order0_128_128_64.pkl')
    # # show3Dslice(np.concatenate([dataset_pre['train_set'][0]['img'],dataset['train_set'][0]['mask']],axis=1))
    # dataset = get_dataset_CIFAR10()
    # dataset = get_dataset_CIFAR100()
    # dataset = get_dataset_imagewoof()
    # dataset = get_dataset_imagenette(preload_cache=True)
    # dataset = get_dataset_imagewoof(preload_cache=True, aim_shape=32)
    # for _ in range(9):
    # #     dataset = list_slide(dataset, 3)
    #     dataset['train_set'] = list_slide(dataset['train_set'], 3)
    #     # dataset.update({'dataset_name':'imagewoof'})
    #     show2D(dataset['train_set'][0]['img'] / 255)






