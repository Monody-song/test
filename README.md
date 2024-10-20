# Darknet-53
YOLOv3 采用了 53 层卷积层作为主干，又被叫做 DarkNet-53，网络结构如下所示:

![image](https://github.com/user-attachments/assets/e973cfd6-8323-4c53-b685-2d6a6da86846)

DarkNet-53 是由卷积层和残差层组成。同时需要注意的是，最后三层 Avgpool、Connected 和 Softmax 层是用来在 ImageNet 数据集上训练分类任务时使用的。

当我们使用 DarkNet-53 作为 YOLOv3 中提取图像特征的主干时，则不再使用最后三层。

