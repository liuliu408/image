#1. faster_rcnn调试过程1！！ 
```
1. 训练
# 2块GPU训练
CUDA_VISIBLE_DEVICES='1,3' python tools/train_net.py \
--cfg configs/gj_2gpu_e2e_faster_rcnn_R-50-FPN.yaml \
OUTPUT_DIR /tmp/detectron-output
# 4块GPU训练
CUDA_VISIBLE_DEVICES='0,1,2,3' python tools/train_net.py \
--cfg configs/gj_4gpu_e2e_faster_rcnn_R-50-FPN.yaml \
OUTPUT_DIR /tmp/detectron-output

2. 测试
CUDA_VISIBLE_DEVICES='1' python tools/test_net.py \
    --cfg configs/gj_2gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    TEST.WEIGHTS  models/model_final.pkl \
    NUM_GPUS 1
    
3.可视化
CUDA_VISIBLE_DEVICES='1' python tools/visualize_results.py \
    --dataset gj_2019_test \
    --detections test/detections.pkl \
    --output-dir test

CUDA_VISIBLE_DEVICES='1' python tools/visualize_results.py  --dataset gj_2019_test --detections test/detections.pkl  --output-dir test
```
![图片标题](https://github.com/liuliu408/image/blob/master/image_notebook/A2.png)
所在赛程：初赛 - A榜  状态/得分: 0.83610700000

##我们退出ssh之后程序继续在后台跑呢？screen帮助你实现梦想！！
```
1.shell窗口输入：screen
2.运行后台程序
CUDA_VISIBLE_DEVICES='0,1,2,3' python tools/train_net.py \
--cfg configs/gj_4gpu_e2e_faster_rcnn_R-50-FPN.yaml \
OUTPUT_DIR /tmp/detectron-output

3.ctrl+A+D        # 暂时断开screen会话
4.screen -ls      # 当前所有的 screen进程
5.screen -r ID    # 指定进入相应ID的进程﻿​
```

#2. retinanet调试过程2！！

```
1. 训练
CUDA_VISIBLE_DEVICES='0,1,2,3' python tools/train_net.py \
--cfg configs/gj_retinanet_R-101-FPN_1x.yaml \
OUTPUT_DIR /tmp/detectron-output

2. 测试
CUDA_VISIBLE_DEVICES='1' python tools/test_net.py \
--cfg configs/gj_retinanet_R-101-FPN_1x.yaml \
 TEST.WEIGHTS models/model_final.pkl \
 NUM_GPUS 1
3.可视化

CUDA_VISIBLE_DEVICES='1,2,3' python tools/visualize_results.py \
 --dataset gj_2019_test \
 --detections test/detections.pkl \
 --output-dir test﻿
 
```
 ![图片标题](https://github.com/liuliu408/image/blob/master/image_notebook/A3.png) 
 所在赛程：初赛 - A榜 状态/得分: 0.78247300000
 
##用 retinanet_R-101-FPN_2x.yaml 配置训练出现问题：(梯度训练爆炸!!)
```
json_stats: {"eta": "2 days, 0:43:17", "fl_fpn3": "0.757426", "fl_fpn4": "0.261459", "fl_fpn5": "0.000389", "fl_fpn6": "0.000001", "fl_fpn7": "0.000000", "iter": 140, "loss": "1.237531", "lr": "0.005200", "mb_qsize": 0, "mem": 7084, "retnet_bg_num": "718639.625000", "retnet_fg_num": "8641.375000", "retnet_loss_bbox_fpn3": "0.130857", "retnet_loss_bbox_fpn4": "0.057250", "retnet_loss_bbox_fpn5": "0.000220", "retnet_loss_bbox_fpn6": "0.000000", "retnet_loss_bbox_fpn7": "0.000000", "time": "1.951894"}
json_stats: {"eta": "2 days, 1:17:41", "fl_fpn3": "0.717649", "fl_fpn4": "0.293217", "fl_fpn5": "0.000620", "fl_fpn6": "0.000001", "fl_fpn7": "0.000000", "iter": 160, "loss": "1.178684", "lr": "0.005467", "mb_qsize": 0, "mem": 7084, "retnet_bg_num": "719637.000000", "retnet_fg_num": "8354.875000", "retnet_loss_bbox_fpn3": "0.105930", "retnet_loss_bbox_fpn4": "0.044704", "retnet_loss_bbox_fpn5": "0.000311", "retnet_loss_bbox_fpn6": "0.000000", "retnet_loss_bbox_fpn7": "0.000000", "time": "1.975307"}
CRITICAL train.py: 101: Loss is NaN
INFO loader.py: 126: Stopping enqueue thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
INFO loader.py: 113: Stopping mini-batch loading thread
Traceback (most recent call last):
  File "tools/train_net.py", line 132, in <module>
    main()
  File "tools/train_net.py", line 114, in main
    checkpoints = detectron.utils.train.train_model()
  File "/home/bobo/liuq/detectron/detectron/utils/train.py", line 89, in train_model
    handle_critical_error(model, 'Loss is NaN')
  File "/home/bobo/liuq/detectron/detectron/utils/train.py", line 103, in handle_critical_error
    raise Exception(msg)
Exception: Loss is NaN
```

解决方案：https://github.com/facebookresearch/Detectron/issues/81
修改学习率，默认是0.01，建议，如果是在预训练的基础上微调训练的话，建议将学习率修改为0.0001
BASE_LR: 0.0001

训练中。。。。。
```
bobo@Xixi:~/liuq/detectron$ CUDA_VISIBLE_DEVICES='2,3' python tools/train_net.py \
--cfg configs/gj_retinanet_R-101-FPN_1x.yaml \
OUTPUT_DIR /tmp/detectron-output

bobo@Xixi:~/liuq/detectron$ screen -ls
There are screens on:
 32138.pts-92.Xixi (2019年03月10日 19时26分54秒) (Detached)
 67045.pts-50.Xixi (2019年03月06日 15时49分23秒) (Detached)
 71513.pts-50.Xixi (2019年03月06日 08时11分43秒) (Detached)
 42133.pts-50.Xixi (2019年03月06日 00时03分34秒) (Detached)
 17737.pts-50.Xixi (2019年03月05日 23时37分56秒) (Detached)
 59405.pts-53.Xixi (2019年03月05日 21时49分36秒) (Attached)
 58953.16504 (2019年03月05日 21时49分07秒) (Attached)
 56645.pts-53.Xixi (2019年03月05日 21时46分43秒) (Detached)
 16504.pts-46.Xixi (2019年03月05日 09时41分47秒) (Detached)
 12133.pts-46.Xixi (2019年03月05日 09时38分06秒) (Detached)
 25848.pts-42.Xixi (2019年03月04日 22时46分08秒) (Detached)
 27349.test_screen (2019年01月31日 10时48分42秒) (Dead ???)
 
bobo@Xixi:~/liuq/detectron$ screen -r 32138﻿​
```
**如何确定是否出现梯度爆炸？----现象**
在训练过程中出现梯度爆炸会伴随一些细微的信号，如：
（1）模型无法从训练数据中获得更新；
（2）模型不稳定，导致更新过程中的损失出现显著变化；
（3）训练过程中，模型的损失变为Nan。

https://blog.csdn.net/u013250416/article/details/81410693
https://blog.csdn.net/hai008007/article/details/79676994
https://blog.csdn.net/qq_16761099/article/details/80455667

    loss为Nan原因
    1.梯度爆炸。梯度变得非常大，使得学习过程难以继续。
    2.不当的损失函数。
    3.不当的输入。
    4.池化层中步长比卷积核的尺寸大。
    
    数据集本身问题
    数据集本身可能标注不准确，引入大量噪声。以图片数据集为例，具体情况可能包括：
    1.在目标检测中，四边形标注框与实际不匹配，标注框的面积过多大于实际目标的面积；
    2.在使用mask rcnn检测目标时，只用四点标注的bounding boxes来生成mask，可能会导致生成的mask不够准确。如果偏差过大，也会引入梯度爆炸；
    3.在场景文字检测中，为了套用已有的检测水平方向物体的目标检测框架，将原有的任意四边形标注框转换为其水平方向的最小外接矩形，也会导致标注框的面积过多大于实际目标的面积；
    4.在场景文字识别中，对于一张完整的图片，一般是根据坐标对单词区域进行裁剪，再将所有的单词区域缩放成相同的大小。一是单词区域裁剪不准确，二是如果缩放尺寸没有选择好，较小的图片放大成过大的尺寸，会使得训练图片非常模糊，引入大量噪声。
    对于数据集本身的问题，带来的梯度爆炸问题，一般需要注意，尽量使得标注准确，除非是进行难样本挖掘操作，一般尽量使用清晰的图片。
    3.  如何解决梯度消失与梯度爆炸？
    用ReLU、Leaky ReLU、PReLU、RReLU、Maxout等替代sigmoid函数。
    用Batch Normalization。
