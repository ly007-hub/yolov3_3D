import pickle
import os
import warnings
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


sep = os.sep


def save_as_pkl(save_path, obj):
    data_output = open(save_path, 'wb')
    pickle.dump(obj, data_output)
    data_output.close()

def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
	读取文件夹中带有固定扩展名的文件
	:param filepath:
	:param expname: 扩展名，如'h5','PNG'
	:return: 文件路径list
	"""

    file_name = os.listdir(filepath)
    file_List = []
    for file in file_name:
        if file.endswith('.' + expname):
            file_List.append(os.path.join(filepath, file))
    return file_List

data_config = dict(

    # todo 数据集路径设置

    # 医学数据集-3D
    # 胰腺（具体名字待补充，CT 的），
    miccai_2018_decathlon_data_root=r'/data/liyi219/pnens_3D_data/v1_data/miccai_2018_decathlon/data',
    miccai_2018_decathlon_data_WW=321,  # 窗宽
    miccai_2018_decathlon_data_WL=123,  # 窗位
    miccai_2018_decathlon_data_aimspace=[0.5, 0.5, 0.8],  # respacing
    # miccai_2018_decathlon_data_aimspace = [1.0,1.0,1.6], # respacing
    # miccai_2018_decathlon_data_aimspace = None, # respacing
    miccai_2018_decathlon_data_aimshape=[64, 64, 64],
    # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64] miccai_2018_decathlon_data_aimshape = [
    #  96,96,48], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64]
    miccai_2018_decathlon_data_cut_bg=False,  # 去掉背景 todo 这个步骤及其消耗时间

    NIH_pancreas_data_root=r'/data/liyi219/pnens_3D_data/v1_data/NIH/data',
    NIH_pancreas_data_WW=321,  # 窗宽
    NIH_pancreas_data_WL=123,  # 窗位
    NIH_pancreas_data_aimspace=[0.5, 0.5, 0.8],  # respacing
    # NIH_pancreas_data_aimspace = [1.0,1.0,1.6], # respacing
    # NIH_pancreas_data_aimspace = None, # respacing
    NIH_pancreas_data_aimshape=[64, 64, 64],  # 最终形状，经过resize和减裁的,todo 这个要自己好好计算
    # NIH_pancreas_data_aimshape = [96,96,48], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算
    NIH_pancreas_data_cut_bg=False,  # 去掉背景
)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ShapeError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

def bbox_scale(bbox, trans_rate):
    """
    因为原点是【0，0】，所以坐标直接缩放即可
    :param bbox: 坐标 [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    :param trans_rate: (dim0r,dim1r,dim2r)
    :return: 注意，这里返回的坐标不是整数，只有显示的时候才是整数，网络预测出来的坐标以及计算loss的坐标都不是整数
    """
    trans_rate = list(trans_rate)
    trans_rate = [trans_rate[0], trans_rate[0], trans_rate[1], trans_rate[1], trans_rate[2], trans_rate[2]]
    trans_rate = np.array(trans_rate)
    return list(np.array(trans_rate) * np.array(bbox))


def connected_domain_3D(image):
    """
    返回3D最大连通域
    :param image: 二值图
    :return:
    """
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    # for idx, i in enumerate(num_list_sorted[1:]):
    #     if area_list[i-1] >= (largest_area//10):
    #         final_label_list.append(i)
    #     else:
    #         break
    output = sitk.GetArrayFromImage(output_ex)

    output = output==final_label_list
    output = output.astype(np.float32)
    # for one_label in num_list:
    #     if  one_label in final_label_list:
    #         continue
    #     x, y, z, w, h, d = stats.GetBoundingBox(one_label)
    #     one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
    #     output[z: z + d, y: y + h, x: x + w] *= one_mask
    #
    # if mask:
    #     output = (output > 0).astype(np.uint8)
    # else:
    #     output = ((output > 0)*255.).astype(np.uint8)
    # output = output.astype(np.float32)
    # output[output == final_label_list] = -1.
    # output = output < 0.1
    # output = output.astype(np.uint8)
    return output

def resize3D(img, aimsize, order = 3):
    """

    :param img: 3D array
    :param aimsize: list, one or three elements, like [256], or [256,56,56]
    :return:
    """
    _shape =img.shape
    if len(aimsize)==1:
        aimsize = [aimsize[0] for _ in range(3)]
    if aimsize[0] is None:
        return zoom(img, (1, aimsize[1] / _shape[1], aimsize[2] / _shape[2]), order=order)  # resample for cube_size
    if aimsize[1] is None:
        return zoom(img, (aimsize[0] / _shape[0], 1, aimsize[2] / _shape[2]), order=order)  # resample for cube_size
    if aimsize[2] is None:
        return zoom(img, (aimsize[0] / _shape[0], aimsize[1] / _shape[1], 1), order=order)  # resample for cube_size
    return zoom(img, (aimsize[0] / _shape[0], aimsize[1] / _shape[1], aimsize[2] / _shape[2]), order=order)  # resample for cube_size

class wama():
    """
    以病人为单位的class
    1) 包含图像与标注
    2）不要有复杂操作或扩增，这些应该另外写代码，否则会占用大量内存？
    3) 包含简单的预处理，如调整窗宽窗位，resampleling


    """
    def __init__(self):
        """
        只支持单肿瘤
        """
        # 可能会用到的一些信息
        self.id = None  # 用来存放病人的id的，字符串形式，如's1','1','patient_X'都可
        # 存储图像的信息
        self.scan = {}  # 字典形式储存数据，如image['CT']=[1,2,3]， 不同模态的图像必须要是配准的！暂时不支持没配准的
        self.spacing = {}  # 字典形式存储数据的,tuple
        self.origin = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息 todo
        self.transfmat = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息
        self.axesOrder = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息

        self.resample_spacing = {}  # tuple, 一旦存在，则表示图像已经经过了resample

        # 储存mask，只需储存图像即可
        self.sementic_mask = {}  # 同上，且要求两者大小匹配，暂时只支持一个病人一个肿瘤（否则在制作bbox的时候会有问题）
        self.bbox = {}  # 将mask取最小外接方阵，或自己手动设置


        # 分patch的操作，在外面进行，反正只要最后可以还原就行了
        # 储存分patch的信息（要考虑分patch出现2D和3D的情况）,分patch的时候记得演示分patch的过程
        # self.is_patched = False # 是否进行了分patch的操作  （每次添加了新的数据、模态、mask，都需要将这个设置为False，之后重新分patch）
        # self.patch_num = {}   # patch的数量
        self.patches = {}  # 直接储存patch到list


    """从NIFTI加载数据系列"""
    def appendImageFromNifti(self, img_type, img_path, printflag = False):
        """
        添加影像
        :param img_type: 自己随便定义，就是个自定义的关键字
        :param img_path:
        :param printflag: 是否打印影像信息
        :return:
        """
        # 首先判断是否已有该模态（img_type）的数据
        if img_type in self.scan.keys():
            warnings.warn(r'alreay has type "' + img_type + r'", now replace it')
        # 读取数据
        scan, spacing, origin, transfmat, axesOrder = readIMG(img_path)
        # 存储到对象
        self.scan[img_type] = scan
        self.spacing[img_type] = spacing
        self.origin[img_type] = origin
        self.transfmat[img_type] = transfmat
        self.axesOrder[img_type] = axesOrder
        if printflag:
            print('img_type:', img_type)
            print('img_shape:', self.scan[img_type].shape)
            print('spacing', self.spacing[img_type])
            print('axesOrder', self.axesOrder[img_type])

        self.resample_spacing[img_type] = None  # 初始化为None

    def appendSementicMaskFromNifti(self, img_type, mask_path):
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 读取mask
        mask, bbox, _, _, _, _  = readIMG(mask_path)
        # 检查形状是否与对应img_type的scan一致
        if mask.shape != self.scan[img_type].shape:
            raise ShapeError(r'shape Shape mismatch error: scan "' + img_type + \
                             r'" shape is' + str(self.scan[img_type].shape)+ \
                             r', but mask shape is '+ str(mask.shape))

        # 将mask存入对象
        self.sementic_mask[img_type] = mask

        # 将bbox存入对象
        self.bbox[img_type] = bbox

    def appendImageAndSementicMaskFromNifti(self, img_type, img_path, mask_path, printflag = False):
        self.appendImageFromNifti(img_type, img_path, printflag)
        self.appendSementicMaskFromNifti(img_type, mask_path)


    """读取数据"""
    # 获取整个图像
    def getImage(self, img_type):
        """

        :param img_type:
        :return:  ndarray of whole_size img
        """
        return deepcopy(self.scan[img_type])

    # 获取整个mask
    def getMask(self, img_type):
        """

        :param img_type:
        :return: ndarray of whole_size mask
        """
        return deepcopy(self.sementic_mask[img_type])

    # 获取bbox内的图像
    def getImagefromBbox(self, img_type, ex_voxels=[0,0,0], ex_mms=None, ex_mode='bbox', aim_shape=None):
        """
        先用mask和原图点乘，之后外扩一定体素的bbox取出来（注意，各个维度外扩的尺寸是固定的，暂时）,
        :param img_type:
        :param ex_voxels: 三个值！不要乱搞乱赋值，ex_voxels = [20,20,20] 的样子
        :param ex_mms: 指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到原图
        mask_roi_img = self.scan[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        # ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=3)

        return roi_img

    # 获取mask内的图像
    def getImagefromMask(self, img_type, ex_voxels=[0,0,0], ex_mms=None, ex_mode ='bbox', aim_shape = None):
        """
        先用mask和原图点乘，之后外扩一定体素的bbox取出来（注意，各个维度外扩的尺寸是固定的，暂时）,
        :param img_type:
        :param ex_voxels: 3个值！不要乱搞乱赋值，ex_voxels = [20,20,20]  这样子
        :param ex_mms: 指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 用mask和原图点乘
        mask_roi_img = self.scan[img_type] * self.sementic_mask[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        # ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=3)

        return roi_img

    # 获取bbox内的mask
    def getMaskfromBbox(self, img_type, ex_voxels=[0,0,0], ex_mms=None, ex_mode='bbox', aim_shape=None):
        """
        外扩一定体素的bbox取出来,
        :param img_type:
        :param ex_voxels: 3个值！不要乱搞乱赋值，ex_voxels = [20,20,20] 这样子
        :param ex_mms: 1个值，指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到mask
        mask_roi_img = self.sementic_mask[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        # ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=0)

        return roi_img


    # 获取bbox（坐标，前提是已经有bbox或者mask）
    def getBbox(self, img_type):
        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到bbox
        bbox = self.bbox[img_type]

        return bbox

    # """从Array加载数据系列"""
    # def appendImageFromArray(self, img_type ,scan, spacing, origin, transfmat, axesOrder):
    #     """
    #
    #     :param img_type: 例如'CT'
    #     :param scan: ndarray，需要时3D array，axis需要和spacing一致
    #     :param spacing:
    #     :param origin:
    #     :param transfmat:
    #     :param axesOrder: 如[coronal,sagittal,axial]，必须和scan、spacing、transfmat的axis一致
    #     :return:
    #     """


    # def appendSementicMaskFromArray(self, img_type, mask_path):
    #     self.shape_check()
    #
    # def appendImageAndSementicMaskFromArray(self, img_type, img_path, mask_path):


    """基于mayavi的可视化"""
    def show_scan(self, img_type, show_type = 'volume'):
        """

        :param img_type:
        :param show_type: volume or slice
        :return:
        """
        # 检查是否存在对应模态
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查是否安装了mayavi
        if mayavi_exist_flag:
            if show_type == 'volume':
                show3D(self.scan[img_type])
            elif show_type == 'slice':
                show3Dslice(self.scan[img_type])
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_mask(self, img_type, show_type = 'volume'):
        """

        :param img_type:
        :param show_type: volume or slice
        :return:
        """
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        if mayavi_exist_flag:
            if show_type == 'volume':
                show3D(self.sementic_mask[img_type])
            elif show_type == 'slice':
                show3Dslice(self.sementic_mask[img_type])
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_MaskAndScan(self, img_type, show_type = 'volume'):
        """
        拼接在一起显示
        :param img_type:
        :param show_type:
        :return:
        """
        # 只检查mask即可，因为有mask必有image
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        if mayavi_exist_flag:
            # 读取mask和image，并拼接
            mask = self.sementic_mask[img_type]
            image = self.scan[img_type]
            image_mask = np.concatenate([mat2gray(image),mat2gray(mask)],axis=1)
            image_mask = image_mask*255

            if show_type == 'volume':
                show3D(image_mask)
            elif show_type == 'slice':
                show3Dslice(image_mask)
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_bbox(self, img_type, line_thick = 2):
        """
        显示bbox，（这里只是简单的显示bbox的形状，并不是在全图显示bbox的位置）
        :param img_type:
        :param show_type:
        :return:
        """
        bbox = self.getBbox(img_type=img_type)
        show3Dbbox(bbox, line_thick = line_thick)

    def show_bbox_with_img(self, img_type, show_type='volume'):
        """
        显示bbox内的图像
        :param img_type:
        :param show_type:
        :return:
        """
        raise NotImplementedError

    """ annotation操作 """
    def make_bbox_from_mask(self, img_type, big_connnection = False):
        """
        目前只支持单肿瘤
        :param img_type:
        big_connnection: 是否基于最大连通域，如果是粗标注，则设为False
        :return:
        """

        # 检查对应的img_type是否有mask
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        # 提取mask
        mask = self.sementic_mask[img_type]

        # 若只取最大连通域，则执行取最大连通域操作
        if big_connnection:
            mask = connected_domain_3D(mask)

        # 计算得到bbox，形式为[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
        indexx = np.where(mask> 0.)
        dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                                np.min(indexx[1]), np.max(indexx[1]),
                                                                np.min(indexx[2]), np.max(indexx[2])]
        self.bbox[img_type] = [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]

    def add_box(self, img_type, bbox):
        """
        ！！ 需要在resample操作前进行，一旦经过了resample，就不可以添加Bbox了（我是不相信你会自己去算😊）
        :param bbox: 要求按照此axis顺序给出  coronal,sagittal,axial （或x,y,z）
                    example ：[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
        """
        # 检查是否有对应img_type的图像
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查坐标是否超出范围
        if checkoutIndex(self.scan[img_type], bbox):
            raise IndexError('Bbox index out of rang')

        # 加入坐标
        self.bbox_mask[img_type] = bbox
        # 利用坐标生成mask， 方便resample的操作
        # 储存mask

    def get_bbox_shape(self, img_type):
        """返回肿瘤的大小: 即lenth_dim0到2， list
            注意，返回voxel number， 同时返回true size（mm^3），（cm^3）
        """
        # 先看看有妹有bbox，有就直接搞出来
        if img_type in self.bbox.keys():
            print('get from bbox')
            bbox = self.bbox[img_type]
            return [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]

        # 妹有就看看有妹有mask，有就直接调用，注意连通域函数
        if img_type in self.sementic_mask.keys():
            # 得到bbox
            print('making bbox')
            self.make_bbox_from_mask(img_type)
            # 返回shape
            print('get from bbox')
            bbox = self.bbox[img_type]
            return [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]

        # 啥都没有，就算了
        warnings.warn(r'you need to load "' + img_type + r'" mask or bbox first')
        return

    def get_scan_shape(self, img_type):
        return self.scan[img_type].shape




    """prepocessing"""
    def resample(self, img_type, aim_spacing, order = 3): # TODO
        """

        :param img_type:
        :param aim_space: tuple with 3 elements (dim0, dim1, dim2), or 1 interger
        :return:
        """
        # 原图、mask、bbox都需要！！！，bbox可以先转为矩阵，然后resize后重新获得坐标

        # 检查是否有对应的image
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查aim_spacing todo

        # 首先计算出各个轴的scale rate （这里要确保scan和spacing的dim是匹配的）
        # 这里需要注意！：如果已经经过了resample，那么需要将最后一此resample的spacing作为当前的spacing todo
        if self.is_resample(img_type):
            or_spacing = self.resample_spacing[img_type]
        else:
            or_spacing = self.spacing[img_type]
        trans_rate = tuple(np.array(or_spacing)/np.array(aim_spacing))

        # resample， 并记录aim_spacing, 以表示图像是经过resample的
        # 记录aim_spacing
        self.resample_spacing[img_type] = aim_spacing
        # 先对原图操作
        self.scan[img_type] = zoom(self.scan[img_type], trans_rate, order=order) # 用双三次插值？
        # 再对mask操作
        if img_type in self.sementic_mask.keys():
            self.sementic_mask[img_type] = zoom(self.sementic_mask[img_type], trans_rate, order=0)  # 最近邻插值？（检查下是不是还是二值图接可）
        # 再对BBox操作（转化为mask，之后resize，之后取bbox）
        if img_type in self.bbox.keys():
            self.bbox[img_type] = bbox_scale(self.bbox[img_type], trans_rate) # todo 需要检查一下

    def is_resample(self, img_type):
        """
        判断图像是否经过resample, 若已经经过重采样，则返回True
        :param img_type:
        :return:
        """
        if  self.resample_spacing[img_type] is not None:
            return True
        else:
            return False

    def adjst_Window(self,img_type, WW, WL):
        """
        调整窗宽窗位
        :param img_type: 图像种类
        :param WW: 窗宽
        :param WL: 窗位
        :return:
        """
        self.scan[img_type] = adjustWindow(self.scan[img_type], WW, WL)

    def slice_neibor_add(self, img_type, axis = ['axial'], add_num = [5], add_weights = 'Gaussian', g_sigma = 3):
        """
        任何时候操作都可以，只能对scan操作
        slice neighbor add, 相邻层累加策略，类似 mitk 里面的那个多层叠加显示的东西，等价于平滑

        指定1个axis， ok ，那么只在这一个axis操作
        如果2个，则各自在1个axis操作，之后2个操作后的矩阵取平均
        3个也同理

        ！！ 直观上讲，最好resample到voxel为正方形再搞，不过实际上是无所谓
        :param img_type:
        :param axis: list, can be ['coronal','sagittal','axial'], ['x','y','z'], [0, 1, 2]
        :param add_num: list, 维度要和axis匹配，且list中的element必须是正奇数
        :param add_weights: ‘Gaussian’，‘Mean’， ‘DeGaussian’（即1-maxminscale（Gaussian））
        :param g_sigma: ‘Gaussian’或‘DeGaussian’模式下的方差，越大权重越接近于mean
        :return:
        """

        # 用来存放各个变换后的图像
        tmp_scan_list = []

        # 逐个完成变换
        for index, _axis in enumerate(axis):
            tmp_scan_list.append(slice_neibor_add_one_dim(self.scan[img_type], _axis, add_num[index], add_weights, g_sigma))
            # tmp_scan_list.append(slice_neibor_add_one_dim(mask, _axis, add_num[index], add_weights, g_sigma))

        # 将变换后的所有图像取平均， 重新赋予
        if len(tmp_scan_list)== 1:
            return tmp_scan_list[0]
        elif len(tmp_scan_list)==2:
            return (tmp_scan_list[0]+tmp_scan_list[1])/2
        elif len(tmp_scan_list)==3:
            return (tmp_scan_list[0]+tmp_scan_list[1]+tmp_scan_list[2])/3

    def _normalization(self, img_type):
        self.scan[img_type] = standardization(self.scan[img_type])

    def _minmaxscale(self, img_type):
        self.scan[img_type] = mat2gray(self.scan[img_type])


    """postprocessing"""
    #这个操作可以挪到外面，因为最后还是要分开保存
    def makePatch(self, mode, **kwargs):
        """
        逻辑：
        1）先框取ROI获得bbox，之后在ROI内进行操作
        2）外扩roi
        3）将roi内图像拿出，缩放到aim_shape
        4）分patch

        参数部分（部分参数和
        :param mode: 'slideWinND'可以当作1D、2D、3D使用   ('windmill'暂不支持，slideWin1D懒得用了反正slideWinND可以代替slideWin1D的功能）
        :param kwargs: 大部分参数与getImagefromBbox一样
            img_type
            slices
            stride
            expand_r
            ex_mode
            ex_voxels
            ex_mms
            aim_shape
        """

        # 从kwargs中获取参数
        img_type = kwargs['img_type']
        slices = kwargs['slices']  # list 包含三个元素，对应三个轴的层数（滑动窗尺寸）
        stride = kwargs['stride']  # list 包含三个元素，对应三个轴的滑动步长
        expand_r = kwargs['expand_r']   # 一般是[1,1,1],类似膨胀卷积，即不膨胀
        ex_mode = kwargs['ex_mode']  # 'bbox' or 'square', bbox则保持之前的形状

        if 'ex_voxels' in kwargs.keys():  # 不指定则默认不外扩，即等于[0,0,0]
            ex_voxels = kwargs['ex_voxels']
        else:
            ex_voxels = [0, 0, 0]

        if 'ex_mms' in kwargs.keys():  # 因为这个不是必须指定的，但是指定了就优先级比ex_voxels高
            ex_mms = kwargs['ex_mms']
        else:
            ex_mms = None

        if 'aim_shape' in kwargs.keys():  # 因为这个不是必须指定的，可以理解为based_shape,patch就是基于这个进行分块的
            aim_shape = kwargs['aim_shape']
        else:
            aim_shape = None  # 保持原来的形状


        # 不同模式开始分patch
        if mode == 'slideWinND':
            # 检查各个参数
            if (len(slices) is not 3 or
                    len(stride) is not 3 or
                    len(expand_r) is not 3 or
                    len(ex_voxels) is not 3):
                raise FormatError('length of slices & stride & expand_r & ex_voxels should be 3')


            # 开始分patch
            patches = slide_window_n_axis(array3D = self.scan[img_type],
                                          spacing=self.spacing[img_type],
                                          origin=self.origin[img_type],
                                          transfmat=self.transfmat[img_type],
                                          axesOrder=self.axesOrder[img_type],
                                          bbox = self.getBbox(img_type=img_type),
                                          slices=slices,
                                          stride=stride,
                                          expand_r=expand_r,
                                          mask=self.sementic_mask[img_type],
                                          ex_mode=ex_mode,
                                          ex_voxels=ex_voxels,
                                          ex_mms=ex_mms,
                                          resample_spacing=self.resample_spacing[img_type],
                                          aim_shape=aim_shape)
        else:
            raise ValueError('mode should be slideWinND')


        self.patches[img_type] = patches

def readIMG(filename):
    """
    read mhd/NIFTI image
    :param filename:
    :return:
    scan 图像，ndarray，注意这里已经改变了axis，返回的图axis对应[coronal,sagittal,axial], [x,y,z]
    spacing：voxelsize，对应[coronal,sagittal,axial], [x,y,z]
    origin：realworld 的origin
    transfmat：方向向量组成的矩阵，一组基向量，3D的话，一般是(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)，也即代表
                [1,0,0],[0,1,0],[0,0,1]三个基向量，分别对应
    """
    itkimage = sitk.ReadImage(filename)
    # 读取图像数据
    scan = sitk.GetArrayFromImage(itkimage) #3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
    scan = np.transpose(scan, (1, 2, 0))     # 改变axis，对应的axis[coronal,sagittal,axial]，即[y，x，z] 冠状面、矢状面、水平面
    bbox = getbbox(scan)
    # 读取图像信息
    spacing = itkimage.GetSpacing()        #voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    axesOrder = ['coronal', 'sagittal', 'axial']  # 调整顺序可以直接axesOrder = [axesOrder[0],axesOrder[2],axesOrder[1]]
    return scan, bbox, spacing, origin, transfmat, axesOrder

def getbbox(scan):

    coronal_bbox = []
    for i in range(len(scan[:, 0, 0])):
        if 1 in scan[i, :, :]:
            coronal_bbox.append(i)

    sagittal_bbox = []
    for i in range(len(scan[0, :, 0])):
        if 1 in scan[:, i, :]:
            sagittal_bbox.append(i)

    axial_bbox = []
    for i in range(len(scan[0, 0, :])):
        if 1 in scan[:, :, i]:
            axial_bbox.append(i)

    bbox = [coronal_bbox[0], coronal_bbox[-1], sagittal_bbox[0], sagittal_bbox[-1], axial_bbox[0], axial_bbox[-1]]

    return bbox