# 检测与跟踪
我们采用 TBD(Tracking by Detection) 的方法在单目视频中取检测跟踪自行车的轨迹.我们使用 **DeepSORT**<sup>[[1]](#1)</sup>作为跟踪器， **YOLOv3**<sup>[[2]](#2)</sup>作为检测器。
其中，YOLOv3经过了迁移学习，类别的输出次序依次为: ***person***,***bicycle***,***car***,***motorcycle***,***bus***,***train***,***truck***.值得注意的是，虽然本项目的研究对象是自行车，但为了扩大本项目的应用范围，提高其实用性，故在训练时把人、摩托车、汽车等常见的道路使用者也纳为检测对象。
这一部分内容主要借鉴了[qqwwee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)和[Qidian213/deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3)等人的工作。
## 主要环境配置
- python 3
- tensorflow 1.9.0
- keras 2.2.4
## 快速使用
该算法主要有三部分组成：检测器，特征提取器和跟踪器，在代码中即 ***detections***,***encoder***,***tracker***. 检测器负责定位目标位置并确定其类型；特征提取器用来是一个轻型的卷积神经网路，用来提取图像上bbox框出区域的特征，为跟踪器服务；跟踪器则完成跟踪任务，匹配相邻两帧中的相同目标。

0. 下载权重：https://pan.baidu.com/s/1pRN6HVifNTISnzZlREzZKg, 最好放在 ***model_data*** 文件夹中
1. 创建 ***YOLO*** 检测器
```python
yolo = YOLO(model_path = 'model_data/yolo_BBA.h5',
            classes_path = 'model_data/classes_name.txt',
            weights_only = True,
            score = 0.3,
            iou = 0.3)
```
2. 创建 ***encoder*** 和 ***tracker***
```python
encoder = gdet.create_box_encoder('model_data/mars-small128.pb',batch_size=1)
tracker = Tracker(metric_mode="cosine",max_cosine_distance=max_cosine_distance,nn_budget=nn_budget)#defaultly max_cosine_distance = 0.3, nn_budget = None
```
3. 检测目标位置并提取器特征
```python
boxs,classes,scores = yolo.detect_image(image)# detect image
features = encoder(frame,boxs)#encoder features
detections = [Detection(bbox, score, feature,class_)
                        for bbox,score,feature,class_ in zip(boxs,scores,features,classes)] # create dectectors
```
4. 更新 ***tracker***
```python
tracker.predict()
tracker.update(detections)#the results are stored in tracker.tracks
```

# 其他资源
1. 测试视频：
    - [Road Bicycle Race](https://pan.baidu.com/s/1FJ1vEFBoucHJTQBrFEWXOQ)
    - [New York City 4K](https://pan.baidu.com/s/15lP9ZhgeIrXm0BZqBG1tiQ)

# 参考文献
<a name="1">[1]</a>: Wojke, Nicolai, Alex Bewley, and Dietrich Paulus. "Simple Online and Realtime Tracking with a Deep Association Metric." Paper presented at the 2017 IEEE International Conference on Image Processing (ICIP), 2017.

<a name="2">[2]</a>: Redmon, Joseph, and Ali Farhadi. "Yolov3: An Incremental Improvement."  2018.
