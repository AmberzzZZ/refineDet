from resnet_back import resnet_back
from vgg_back import vgg16_back300, vgg16_back512
from keras.layers import Input, Lambda, Conv2D, ReLU, add, Deconv2D, Lambda
from keras.models import Model


def refineDet(input_shape=(512,512,3), back='resnet', n_classes=20, n_anchors=9):

    inpt = Input(input_shape)

    if back == 'resnet':
        arm_sources = resnet_back(inpt, depth=101)
    if back == 'vgg':
        arm_sources = vgg16_back512(inpt)

    odm_sources = fpn(arm_sources)

    # rpn head
    rpn_cls_outputs, rpn_box_outputs = rpn_head(arm_sources, n_anchors)

    # refine head
    refine_cls_outputs, refine_box_outputs = refine_head(odm_sources, n_classes, n_anchors)

    # rpn_loss_ = Lambda(rpn_loss)([*rpn_cls_outputs, *rpn_box_outputs])
    # refine_loss_ = Lambda(refine_loss)([*refine_cls_outputs, *refine_box_outputs])

    model = Model(inpt, [*rpn_cls_outputs, *rpn_box_outputs, *refine_box_outputs, *refine_box_outputs])

    return model


def rpn_head(arm_sources, n_anchors=9):
    rpn_cls_outputs = []
    rpn_box_outputs = []
    for idx, feature in enumerate(arm_sources):
        x = Conv2D(1*n_anchors, 3, strides=1, padding='same', name='rpn_cls_%d' % idx)(feature)
        rpn_cls_outputs.append(x)
        x = Conv2D(4*n_anchors, 3, strides=1, padding='same', name='rpn_box_%d' % idx)(feature)
        rpn_box_outputs.append(x)
    return rpn_cls_outputs, rpn_box_outputs


def refine_head(odm_sources, n_classes, n_anchors=9):
    refine_cls_outputs = []
    refine_box_outputs = []
    for idx, feature in enumerate(odm_sources):
        x = Conv2D(n_classes*n_anchors, 3, strides=1, padding='same', name='refine_cls_%d' % idx)(feature)
        refine_cls_outputs.append(x)
        x = Conv2D(4*n_anchors, 3, strides=1, padding='same', name='refine_box_%d' % idx)(feature)
        refine_box_outputs.append(x)
    return refine_cls_outputs, refine_box_outputs


def fpn(features, fpn_channel=[512, 1024, 512, 256]):

    odm_4, up4 = tc_block(features[-1], None, fpn_channel[-1])
    odm_3, up3 = tc_block(features[-2], up4, fpn_channel[-2])
    odm_2, up2 = tc_block(features[-3], up3, fpn_channel[-3])
    odm_1, _ = tc_block(features[-4], up2, fpn_channel[-4])

    return [odm_1, odm_2, odm_3, odm_4]


def tc_block(forward, up, n_filters):

    forward = Conv2D(n_filters, 3, strides=1, padding='same', activation='relu')(forward)
    forward = Conv2D(n_filters, 3, strides=1, padding='same', activation=None)(forward)

    if up is None:
        up = forward
    else:
        up = Deconv2D(n_filters, 4, strides=2, padding='same')(up)
        up = add([forward, up])

    forward = ReLU()(up)
    forward = Conv2D(n_filters, 3, strides=1, padding='same', activation='relu')(forward)

    return forward, up


if __name__ == '__main__':
    model = refineDet(back='resnet')
    print(model.outputs)
    # model.summary()






