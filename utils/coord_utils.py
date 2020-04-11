import numpy as np
from scipy.interpolate import splev, splrep
import scipy.signal as signal

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

def CSI(x,y):
    '''
    计算轨迹曲率
    ================
    Parameters:
        x:数据点的横坐标构成的向量
        y:数据点的纵坐标构成的向量
    ================
    Returns:
        cur:每个数据点的曲率
    '''
    if not len(x)==len(y):
        raise ValueError('Dimension cannot match')
    N=len(x)
    if N<=1:
        return 0
        # raise ValueError('More points are needed')
    #初始化矩阵
    D2=np.zeros(N)
    T=np.zeros((N,N))
    S=np.zeros(N)
    h=np.array([x[k+1]-x[k] for k in range(0,N-1)])
    #矩阵赋值
    T[0][0]=1
    for k in range(1,N-1):
        T[k][k-1]=h[k-1]
        T[k][k]=2*(h[k-1]+h[k])
        T[k][k+1]=h[k]
        S[k]=6*(y[k+1]-y[k])/h[k]-6*(y[k]-y[k-1])/h[k-1]
    T[N-1][N-2]=1
    T[N-1][N-1]=2
    #解线性方程组，m是二阶导数，b是一阶导数
    m=np.linalg.solve(T,S)
    b=[(y[k+1]-y[k])/h[k]-h[k]*m[k]/2-h[k]*(m[k+1]-m[k])/6 for k in range(0,N-1)]
    b.append((y[N-1]-y[N-2])/h[N-2])
    #输出每个点对应的曲率
    cur=np.array([abs(m[k])/(1+b[k]**2)**(1.5) for k in range(0,N)])
    #输出一个曲率向量
    return cur

def computeVelocity(X,Y,Z,T=1/30):
    """
    由离散点三维坐标计算其速度
    ================
    Parameters:
        X: X坐标,np.array
        Y: Y坐标,np.array
        Z: Z坐标,np.array
        T: 时间间隔,np.array or float
    ================
    Returns:
        vx: x方向的速度，np.array
        vy: y方向的速度，np.array
        vz: z方向的速度，np.array
        v: 速度绝对值，np.array
    """

    X = np.array(X); Y = np.array(Y); Z = np.array(Z)
    if not X.size==Y.size==Z.size:
        raise Exception("The length of X、Y、Z should be equal.")
    if X.size <= 1:
        return 0,0,0,0

    vx = np.diff(X, n=1)
    vy = np.diff(Y, n=1)
    vz = np.diff(Z, n=1)
    if type(T) == float:
        vx = vx / T
        vy = vy / T
        vz = vz / T
    elif type(T) == np.ndarray:
        if not X.size==Y.size==Z.size==T.size:
            raise Exception("The length of X、Y、Z and T should be equal.")
        dT = np.diff(T, n=1)
        vx = vx / dT
        vy = vy / dT
        vz = vz / dT
    else:
        raise Exception("type(T) should be float or np.ndarray")

    # 滤波处理
    vx = medfiltAndSpline(vx);vy = medfiltAndSpline(vy);vz = medfiltAndSpline(vz);
    # 补齐
    vx = np.hstack([vx[0],vx]);vy = np.hstack([vy[0],vy]);vz = np.hstack([vz[0],vz])
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return vx,vy,vz,v

def computeAcceleration(X,Y,Z,T=1/30):
    """
    由离散点三维坐标计算其加速度
    ================
    Parameters:
        X: X坐标,np.array
        Y: Y坐标,np.array
        Z: Z坐标,np.array
        dT: 时间间隔,np.array or float
    ================
    Returns:
        ax: x方向的加速度，np.array
        ay: y方向的加速度，np.array
        az: z方向的加速度，np.array
        a: 加速度绝对值，np.array
    """
    if not X.size==Y.size==Z.size:
        raise Exception("The length of X、Y、Z should be equal.")

    if X.size <= 1:
        return 0,0,0,0

    vx,vy,vz,v = computeVelocity(X,Y,Z,T)
    ax = np.diff(vx, n=1)
    ay = np.diff(vy, n=1)
    az = np.diff(vz, n=1)
    if type(T) == float:
        ax = ax/T
        ay = ay/T
        az = az/T
    elif type(T) == np.ndarray:
        if not X.size==Y.size==Z.size==T.size:
            raise Exception("The length of X、Y、Z and T should be equal.")
        dT = np.diff(T, n=1)
        ax = ax/dT
        ay = ay/dT
        az = az/dT
    else:
        raise Exception("type(T) should be float or np.ndarray")

    dv = np.diff(v, n=1)

    ax = medfiltAndSpline(ax);ay = medfiltAndSpline(ay);az = medfiltAndSpline(az);
    ax = np.hstack([ax[0],ax])#补齐
    ay = np.hstack([ay[0],ay])
    az = np.hstack([az[0],az])

    a = np.sqrt(ax**2 + ay**2 + az**2)
    for i in range(dv.size):
        if dv[i]<0:
            a[i] = -a[i]

    return ax,ay,az,a

def medfiltAndSpline(v,kernel_size=5,s=5):
    '''
    中值滤波后用样条函数拟合
    =============
    Parameters:
        v: 要处理的向量,np.array
        kernel_size: 中值滤波的核大小
        s: 样条函数的拟合程度
    =============
    Returns:
        v_filt: 处理后的向量
    '''
    if len(v)<=5:
        return v
    v_md = signal.medfilt(volume=v, kernel_size=kernel_size)
    spl = splrep(range(len(v_md)), v_md,s=s)
    v_filt = splev(range(len(v_md)), spl)

    return v_filt
