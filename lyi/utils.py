import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pickle

try:
    from mayavi import mlab

    print('mayavi already imported')
    mayavi_exist_flag = True
except:
    print('no mayavi')
    mayavi_exist_flag = 0


def getbbox2D(scan):
    # input : [冠状面coronal, 矢状面sagittal, 水平面axial][x, y, z]
    bbox = []
    for i in range(scan.shape[2]):
        coronal_bbox = []
        sagittal_bbox = []
        if 1 in scan[:, :, i]:
            for j in range(scan.shape[0]):
                if 1 in scan[j, :, i]:
                    coronal_bbox.append(j)
            for j in range(scan.shape[1]):
                if 1 in scan[:, j, i]:
                    sagittal_bbox.append(j)
        else:
            coronal_bbox.append(0)
            sagittal_bbox.append(0)

        bbox.append([coronal_bbox[0], coronal_bbox[-1], sagittal_bbox[0], sagittal_bbox[-1]])
    # [x1 ,x2, y1, y2] [水平面][冠状面， 矢状面]

    return bbox


def getbbox3D(scan):
    # [冠状面coronal, 矢状面sagittal, 水平面axial][x, y, z]
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
    # [x1 ,x2, y1, y2, z1, z2][冠状面， 矢状面， 水平面]
    bbox = [coronal_bbox[0], coronal_bbox[-1], sagittal_bbox[0], sagittal_bbox[-1], axial_bbox[0], axial_bbox[-1]]

    return bbox


def show2D(img2D):
    plt.imshow(img2D, cmap=plt.cm.gray)
    plt.show()


def show2D_z(img2D, z):
    show2D(img2D[:, :, z])


def show3Dslice(image):
    """
    查看3D体，切片模式
    :param image:
    :return:
    """
    mlab.volume_slice(image, colormap='gray',
                      plane_orientation='x_axes', slice_index=10)  # 设定x轴切面
    mlab.volume_slice(image, colormap='gray',
                      plane_orientation='y_axes', slice_index=10)  # 设定y轴切面
    mlab.volume_slice(image, colormap='gray',
                      plane_orientation='z_axes', slice_index=10)  # 设定z轴切面
    mlab.colorbar(orientation='vertical')
    mlab.show()


def show3D(img3D):
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
    mlab.colorbar(orientation='vertical')
    mlab.show()


def show3Dbbox(bbox3D, line_thick=None, img=None):
    """
    粗略的看下bbox的现状
    :param bbox3D: list，6 elements，形如[1,60,1,70,1,80]
    :return:
    """
    # 构建一个稍微比bbox
    if img==None:
        tmp_img = np.zeros([bbox3D[1] - bbox3D[0],
                            bbox3D[3] - bbox3D[2],
                            bbox3D[5] - bbox3D[4]], dtype=np.int)
    else:
        tmp_img = img
    if line_thick is None:
        line_thick = np.max(tmp_img.shape) // 20

    tmp_img[0:line_thick, :, 0:line_thick] = 100
    tmp_img[:, 0:line_thick, 0:line_thick] = 100
    tmp_img[0:line_thick, 0:line_thick, :] = 100

    tmp_img[-1:-(line_thick + 1):-1, :, 0:line_thick] = 100
    tmp_img[:, -1:-(line_thick + 1):-1, 0:line_thick] = 100
    tmp_img[0:line_thick, -1:-(line_thick + 1):-1, :] = 100

    tmp_img[-1:-(line_thick + 1):-1, :, -1:-(line_thick + 1):-1] = 100
    tmp_img[:, -1:-(line_thick + 1):-1, -1:-(line_thick + 1):-1] = 100
    tmp_img[-1:-(line_thick + 1):-1, -1:-(line_thick + 1):-1, :] = 100

    tmp_img[0:line_thick, :, -1:-(line_thick + 1):-1] = 100
    tmp_img[:, 0:line_thick, -1:-(line_thick + 1):-1] = 100
    tmp_img[-1:-(line_thick + 1):-1, 0:line_thick, :] = 100

    show3D(tmp_img)


def bbox_in_img_for_slice(img, bbox, line_thick=1, line_value=-2e3):
    """
    @warning img和bbox对应维度一致, xyz维度顺序不影响
    @param eg img: x, y, z
    @param eg bbox: x1 y1 z1 x2 y2 z2
    @param line_thick: 画线宽度
    @param line_value: 画线颜色深浅
    @return: img 和 bbox 的融合矩阵
    """
    if False:
        img, bbox = img3D, bbox3D
        line_thick = 2
    x1, y1, z1, x2, y2, z2 = bbox
    tmp_img = img

    # x轴方向四条线
    tmp_img[x1:x2, y1:y1 + line_thick, z1:z1 + line_thick] = line_value
    tmp_img[x1:x2, y2 - line_thick:y2, z1:z1 + line_thick] = line_value
    tmp_img[x1:x2, y2 - line_thick:y2, z2 - line_thick:z2] = line_value
    tmp_img[x1:x2, y1:y1 + line_thick, z2 - line_thick:z2] = line_value

    # y轴方向四条线
    tmp_img[x1:x1 + line_thick, y1:y2, z1:z1 + line_thick] = line_value
    tmp_img[x2 - line_thick:x2, y1:y2, z1:z1 + line_thick] = line_value
    tmp_img[x2 - line_thick:x2, y1:y2, z2 - line_thick:z2] = line_value
    tmp_img[x1:x1 + line_thick, y1:y2, z2 - line_thick:z2] = line_value

    # z轴方向四条线
    tmp_img[x1:x1 + line_thick, y1:y1 + line_thick, z1:z2] = line_value
    tmp_img[x1:x1 + line_thick, y2 - line_thick:y2, z1:z2] = line_value
    tmp_img[x2 - line_thick:x2, y2 - line_thick:y2, z1:z2] = line_value
    tmp_img[x2 - line_thick:x2, y1:y1 + line_thick, z1:z2] = line_value

    # 四个面
    tmp_img[x1:x2, y1:y1 + line_thick, z1:z2] = line_value
    tmp_img[x1:x2, y2 - line_thick:y2, z1:z2] = line_value
    tmp_img[x1:x1 + line_thick, y1:y2, z1:z2] = line_value
    tmp_img[x2 - line_thick:x2, y1:y2, z1:z2] = line_value

    return tmp_img
def bbox_in_img_for_3D(img, bbox, line_thick=1, line_value=-2e3):
    """
    @warning img和bbox对应维度一致, xyz维度顺序不影响
    @param eg img: x, y, z
    @param eg bbox: x1 y1 z1 x2 y2 z2
    @param line_thick: 画线宽度
    @param line_value: 画线颜色深浅
    @return: img 和 bbox 的融合矩阵
    """
    if False:
        img, bbox = img3D, bbox3D
        line_thick = 2
    x1, y1, z1, x2, y2, z2 = bbox
    tmp_img = img

    # x轴方向四条线
    tmp_img[x1:x2, y1:y1 + line_thick, z1:z1 + line_thick] = line_value
    tmp_img[x1:x2, y2 - line_thick:y2, z1:z1 + line_thick] = line_value
    tmp_img[x1:x2, y2 - line_thick:y2, z2 - line_thick:z2] = line_value
    tmp_img[x1:x2, y1:y1 + line_thick, z2 - line_thick:z2] = line_value

    # y轴方向四条线
    tmp_img[x1:x1 + line_thick, y1:y2, z1:z1 + line_thick] = line_value
    tmp_img[x2 - line_thick:x2, y1:y2, z1:z1 + line_thick] = line_value
    tmp_img[x2 - line_thick:x2, y1:y2, z2 - line_thick:z2] = line_value
    tmp_img[x1:x1 + line_thick, y1:y2, z2 - line_thick:z2] = line_value

    # z轴方向四条线
    tmp_img[x1:x1 + line_thick, y1:y1 + line_thick, z1:z2] = line_value
    tmp_img[x1:x1 + line_thick, y2 - line_thick:y2, z1:z2] = line_value
    tmp_img[x2 - line_thick:x2, y2 - line_thick:y2, z1:z2] = line_value
    tmp_img[x2 - line_thick:x2, y1:y1 + line_thick, z1:z2] = line_value

    return tmp_img


def show3D(img3D):
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
    mlab.colorbar(orientation='vertical')
    mlab.show()


def readIMG(filename):
    """
    read mhd/NIFTI image
    :param filename:
    :return:
    scan 图像，ndarray，注意这里已经改变了axis，返回的图axis对应[coronal,sagittal,axial], [x,y,z]
    spacing：voxelsize，对应[coronal,sagittal,axial], [x,y,z]
    origin：realworld 的origin
    transfmat：方向向量组成的矩阵，一组基向量，3D的话，一般是(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)，也即代表
                [1,0,0],[0,1,0],[0,0,1]三个基向量??读出来不对
    """
    itkimage = sitk.ReadImage(filename)
    # 读取图像数据
    scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal],[z，y，x]
    scan = np.transpose(scan, (1, 2, 0))  # 改变axis，对应的axis[coronal,sagittal,axial]，即[y，x，z]
    # 读取图像信息
    spacing = itkimage.GetSpacing()  # 两个像素之间的间隔，对应的axis[sagittal,coronal,axial]，即[x, y, z]
    origin = itkimage.GetOrigin()  # 世界坐标原点
    transfmat = itkimage.GetDirection()  # 图像本身坐标系相对于世界坐标系的角度余弦，用于改变图像坐标方向
    axesOrder = ['coronal', 'sagittal',
                 'axial']  # [y, x, z]调整顺序可以直接axesOrder = [axesOrder[0],axesOrder[2],axesOrder[1]]

    return scan, spacing, origin, transfmat, axesOrder


def read_nii(filename):
    scan, spacing, origin, transfmat, axesOrder = readIMG(filename)
    return scan


def save_as_pkl(save_path, obj):
    data_output = open(save_path, 'wb')
    pickle.dump(obj, data_output)
    data_output.close()


def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data
