# dataset 的设置
# 比如各个3D数据集的预处理参数等等


config = dict(

    # todo 数据集路径设置

    # 医学数据集-3D
    # 胰腺（具体名字待补充，CT 的），
    miccai_2018_decathlon_data_root=r'/data/liyi219/pnens_3D_data/v1_data/miccai_2018_decathlon',
    miccai_2018_decathlon_data_WW=321,  # 窗宽
    miccai_2018_decathlon_data_WL=123,  # 窗位
    miccai_2018_decathlon_data_aimspace=[0.5, 0.5, 0.8],  # respacing
    # miccai_2018_decathlon_data_aimspace = [1.0,1.0,1.6], # respacing
    # miccai_2018_decathlon_data_aimspace = None, # respacing
    miccai_2018_decathlon_data_aimshape=[128, 128, 64],
    # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64] miccai_2018_decathlon_data_aimshape = [
    #  96,96,48], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64]
    miccai_2018_decathlon_data_cut_bg=False,  # 去掉背景 todo 这个步骤及其消耗时间

    NIH_pancreas_data_root=r'/data/liyi219/pnens_3D_data/v1_data/NIH',
    NIH_pancreas_data_WW=321,  # 窗宽
    NIH_pancreas_data_WL=123,  # 窗位
    NIH_pancreas_data_aimspace=[0.5, 0.5, 0.8],  # respacing
    # NIH_pancreas_data_aimspace = [1.0,1.0,1.6], # respacing
    # NIH_pancreas_data_aimspace = None, # respacing
    NIH_pancreas_data_aimshape=[128, 128, 64],  # 最终形状，经过resize和减裁的,todo 这个要自己好好计算
    NIH_pancreas_data_cut_bg=False,  # 去掉背景
)
