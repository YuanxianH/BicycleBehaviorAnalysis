# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from utils.data_utils import load_frametime,load_fuseddata
from utils.coord_utils import linear_interpolation
from math import *
import colorsys

# 导入相对坐标轨迹
with open("result/tracking_result.json","r") as f:
    tracks_json = json.load(f)

dir_path = "/media/yxhuang/database/binocular_video/"
# dir_path = "G:/binocular_video/"
# 导入拍摄时间
frame_time = load_frametime(dir_path+"20191022_022500_time.txt")
# 导入融合数据
fused_data = load_fuseddata(dir_path+"fused_pose_planePoint_1022_022500.txt")

# 导入轨迹
with open("result/tracking_result_w.json","r") as f:
    tracks = json.load(f)
# 类型名
with open("yolo3_deepsort/model_data/classes_name.txt") as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]
# 生成每种类型对应的颜色
hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
idx_color = {}
for i,cat in enumerate(class_names):
    idx_color[cat] = int(i)

scalex = 1195450
scaley = 3376830
X = np.array(fused_data["x"])-scalex
Y = np.array(fused_data["y"])-scaley
Z = np.array(fused_data["z"])
plt.figure(figsize=(10,6))
plt.plot(X,Y,color=(0,0,0),ls="--",marker=">",label="camera",markersize=5,markevery=0.2)#车辆自身轨迹
# plt.scatter(X,Y,color=(0,0,0),s=1,label="camera")
# 绘制目标轨迹
for track in tracks:
    # if track["track_id"] not in [430]:
    #     continue
    XYZ_w = np.squeeze(np.array(track["XYZ_w"]))
    if len(XYZ_w) != 0:
        XYZ_w = XYZ_w.reshape(XYZ_w.size//3,3)
#         print(XYZ_w)
        X_w = XYZ_w[:,0]-scalex
        Y_w = XYZ_w[:,1]-scaley
        plt.scatter(X_w,Y_w,s=1,color=colors[idx_color[track["category"]]])

for cat in class_names:
    plt.scatter([0],[0],s=10,color=colors[idx_color[cat]],label=cat)
    # plt.plot([0,0.1],[0,0.1],color=colors[idx_color[cat]],label=cat)

plt.title("Trace of Objects and Camera")
plt.xlabel("X(m)")
plt.ylabel("Y(m)")
plt.legend()
plt.show()
