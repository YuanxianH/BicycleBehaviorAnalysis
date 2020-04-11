# vim: expandtab:ts=4:sw=4
import numpy as np
from . import preprocessing

_defaultXYZ = np.reshape(np.array([999,999,999]),(3,1))

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    object_class: string
        category
    feature : array_like
        A feature vector that describes the object contained in this image.
    XYZ : (3,1) np.array
        The 3d coordination of object
    frame : int
        current frame number

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, object_class,info_dict={},feature=None,**kwargs):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.object_class = object_class
        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = []
        # self.feature = np.asarray(feature, dtype=np.float32) if (feature is not None) else []
        self.XYZ = _defaultXYZ

        self.frame = 0
        self.time = 0
        self.distance = 0

        self.__dict__.update(info_dict)
        self.__dict__.update(kwargs)
        assert np.array(self.XYZ).size == 3
        self.XYZ = np.reshape(np.array(self.XYZ),(3,1))

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
