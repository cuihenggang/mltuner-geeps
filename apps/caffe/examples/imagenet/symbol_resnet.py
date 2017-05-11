'''
Reproducing parper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''

import mxnet_converter as mxc

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mxc.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mxc.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mxc.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mxc.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mxc.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mxc.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mxc.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mxc.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mxc.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mxc.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        fused = mxc.Add(
            name = name + '_fused',
            bottoms = [conv3, shortcut])
        return fused
    else:
        bn1 = mxc.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mxc.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mxc.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mxc.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mxc.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mxc.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mxc.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        fused = mxc.Add(
            name = name + '_fused',
            bottoms = [conv2, shortcut])
        return fused

def resnet(units, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    # data = mxc.Variable(name='data')
    data = 'data'
    data = mxc.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mxc.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mxc.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mxc.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mxc.Activation(data=body, act_type='relu', name='relu0')
        body = mxc.Pooling(data=body, name="pool0", kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mxc.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mxc.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    # pool1 = mxc.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    pool1 = mxc.Pooling(data=relu1, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mxc.Flatten(data=pool1, name='flatten')
    fc1 = mxc.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    # return mxc.SoftmaxOutput(data=fc1, name='softmax')

depth = 34
if depth == 18:
  units = [2, 2, 2, 2]
elif depth == 34:
  units = [3, 4, 6, 3]
elif depth == 50:
  units = [3, 4, 6, 3]
elif depth == 101:
  units = [3, 4, 23, 3]
elif depth == 152:
  units = [3, 8, 36, 3]
elif depth == 200:
  units = [3, 24, 36, 3]
elif depth == 269:
  units = [3, 30, 48, 8]
else:
  units = []
num_classes = 1000
bn_mom = 0.9
workspace = 512
memonger = False
resnet(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if depth >=50
                        else [64, 64, 128, 256, 512], num_class=num_classes, data_type="imagenet", bottle_neck = True
                        if depth >= 50 else False, bn_mom=bn_mom, workspace=workspace,
                        memonger=memonger)