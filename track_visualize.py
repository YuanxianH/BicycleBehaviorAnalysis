import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with open("result/tracking_result.json","r") as f:
    tracks_json = json.load(f)

fig = plt.figure()
ax = fig.gca(projection='3d')
track = tracks_json[1648]
ax.scatter(track["XYZs"][0], track["XYZs"][1], track["XYZs"][2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()
