### offical repo: https://github.com/sfzhang15/RefineDet
### 3rd pytorch repo: https://github.com/yqyao/SSD_Pytorch


### back
    vgg16 & resnet101

    extra tails: 
        vgg back直接抄的ssd，保留原来back中的x8 convblock4 feature，剩下的全是extra得到
        resnet back类似，保留原来back中的x8、x16的convblock3、4 feature，剩下的全是extra得到
        extra中主要是一系列的bottleneck，conv-relu构成

    l2Norm in vgg

    


### architecture
    整体上看，第一行就是基于ssd的rpn
    arm和odm之间就是fpn，用TCB替代了custom fpn中的successive conv block
    upsamp用了deconv, 4x4, s2

    individual heads: 3x3 conv
    cls: [b,h,w,a*c], c=1/n_classes
    reg: [b,h,w,a*4]






