# vim: expandtab:ts=4:sw=4
import numpy as np
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.coord_utils import *

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

class MotionState:
    """
    Pending: 待定
    Turning: 变向
    Speedup: 加速
    Slowdown: 减速
    UniformLinear: 匀速直线运动
    """
    Pending = "Pending"
    Turning = "Turning"
    Speedup = "Speed up"
    Slowdown = "Slow down"
    UniformLinear = "Uniform Linear"

class SecurityState:
    """
    Dangerous: 危险
    Caution: 小心
    Safe: 安全
    """
    Dangerous = "Dangerous"
    Caution = "Caution"
    Safe = "Safe"


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    object_class: string
    frame_begin: int
        The frame number when initialize

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    XYZ_array: (3,n) numpy array
        The 3d track of the object
    frame _array:
        Record the frame number when update
    cat_history:
        Record the category of the object
    """

    _defaults = {
        "montion_buffer": 15,
        "cat_bufffer": 25
    }

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                feature = None,**kwargs):
        self.__dict__.update(self._defaults)

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self._n_init = n_init
        self._max_age = max_age
        self.object_class = None

        self.XYZ_array = np.array([])
        self.frame_array = []
        self.time_array = []
        self.features = []
        self.distance = 0
        self.cat_history = []

        self.state = TrackState.Tentative
        self.motion_state = MotionState.Pending
        self.security_state = SecurityState.Caution

        if feature is not None:
            self.features.append(feature)

        self.hits = 1
        self.age = 1
        self.time_since_update =0
        self.velocity = 0
        self.acceleration = 0
        self.curvature = 0

        self.__dict__.update(kwargs)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        !!! Wrong Actually ltwh

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        self.frame_array.append(detection.frame)
        self.time_array.append(detection.time)
        self.cat_history.append(detection.object_class)
        self.cat_history = self.cat_history[-self.cat_bufffer:]
        self.object_class = max(self.cat_history, key=self.cat_history.count)

        if len(self.XYZ_array) == 0:
            self.XYZ_array = detection.XYZ
        else:
            self.XYZ_array = np.hstack([self.XYZ_array,detection.XYZ])

        self.distance = detection.distance
        # compute velocity
        T = self.time_array[-self.montion_buffer:]
        X = self.XYZ_array[0,-self.montion_buffer:]
        if len(set(T)) == len(T):# 没有重复的点
            Y = self.XYZ_array[1,-self.montion_buffer:]
            Z = self.XYZ_array[2,-self.montion_buffer:]

            _,_,_,self.velocity = computeVelocity(X,Y,Z,np.array(T))
            _,_,_,self.acceleration = computeAcceleration(X,Y,Z,np.array(T))
            self.curvature = CSI(X,Y)

        self._update_security_state()
        self._update_motion_state()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def _update_security_state(self):
        """更新安全性"""
        return 0

    def _update_motion_state(self):
        """更新运动状态"""
        return 0
