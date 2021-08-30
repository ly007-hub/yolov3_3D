# config.py

train_cfg = {
    'lr_epoch': (150, 200),
    'max_epoch': 200
}

# yolo_v3 for NIHpnens
anchor_size_3D_try = [[22, 39, 23], [22, 39, 23], [22, 39, 23],
                     [22, 39, 23], [22, 39, 23], [22, 39, 23],
                      [22, 39, 23], [22, 39, 23], [22, 39, 23]]


IGNORE_THRESH = 0.1
