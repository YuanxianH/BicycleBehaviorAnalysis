def linear_interpolation(fused_data,t):
    '''
    线性内插
    =============
    parameters:
        fused_data: dict with values(list)
            { time: ms
              x,y,z: coordinate(m)
              yaw,roll,pitch: pose(rad)
              speed: m/s
              infer_duration:
              infer_method:
              trace_type:
            }
        t: 当前时间(UNIX)
    =============
    return:
        inter_data: dict
            { time: ms
              x,y,z: coordinate(m)
              yaw,roll,pitch: pose(rad)
              speed: m/s
            }
    '''

    m = len(fused_data["time"])#观测数
    idx = 0
    inter_data ={}

    for i in range(m):
        idx = i
        if t >= fused_data["time"][i] and t <= fused_data["time"][i+1] or i == m-2:
            break

    #线性内插
    def _linear_interpolate(t0,t1,t2,y1,y2):
        y0 = (y2-y1)/(t2-t1) * (t0-t1) + y1
        return y0

    t1 = fused_data["time"][idx]
    t2 = fused_data["time"][idx+1]

    inter_data["time"] = t
    inter_data["x"] = _linear_interpolate(t,t1,t2,fused_data["x"][idx],fused_data["x"][idx+1])
    inter_data["y"] = _linear_interpolate(t,t1,t2,fused_data["y"][idx],fused_data["y"][idx+1])
    inter_data["z"] = _linear_interpolate(t,t1,t2,fused_data["z"][idx],fused_data["z"][idx+1])
    inter_data["yaw"] = _linear_interpolate(t,t1,t2,fused_data["yaw"][idx],fused_data["yaw"][idx+1])
    inter_data["roll"] = _linear_interpolate(t,t1,t2,fused_data["roll"][idx],fused_data["roll"][idx+1])
    inter_data["pitch"] = _linear_interpolate(t,t1,t2,fused_data["pitch"][idx],fused_data["pitch"][idx+1])
    inter_data["speed"] = _linear_interpolate(t,t1,t2,fused_data["speed"][idx],fused_data["speed"][idx+1])

    return inter_data
