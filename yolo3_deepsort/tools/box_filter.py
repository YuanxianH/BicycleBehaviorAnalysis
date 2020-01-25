# vim: expandtab:ts=4:sw=4
import numpy as np

def non_max_suppression_idx(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick

def remove_edge_idx(boxes,thickness,image_size):
    '''Remove boxes in the edge and with small area and return indices.

    Parameters:
    ============
    boxes: (l,t,r,b)
    thickness: the thickness of the edge
    image_size: (weight, height)

    Returns:
    ========
    idx: indices of picked boxes,list
    '''
    idx = []
    W,H = image_size
    for i,box in enumerate(boxes):
        l,t,w,h = box
        # 去除位于图像边缘的目标
        if (l<thickness or t<thickness or l+w>W-thickness or t+h>H-thickness)and (w*h<thickness**2):
            continue
        else:
            idx.append(i)
    return idx

def select_COI_idx(classes,COI):
    '''Select classes of interst and return indices.

    Parameters:
    ============
    classes: list of string
    COI: classes of interst, a set

    Returns:
    ========
    idx: indices of picked classes,list
    '''
    idx = []
    for i,c in enumerate(classes):
        if c in COI:
            idx.append(i)
    return idx

def get_by_index(idx,boxes,classes,scores):
    '''Get new boxes, classes and scores by the same idx

    Parameters:
    ===========
    idx: array
    boxes: list of array
    classes: list of string
    scores: list of float

    Returns:
    bboxes,cclasses,sscores: list
    '''
    bboxes = [boxes[i] for i in idx]
    cclasses = [classes[i] for i in idx]
    sscores = [scores[i] for i in idx]

    return bboxes,cclasses,sscores

def remove_edge(boxes,classes,scores,thickness,image_size):
    '''Remove boxes in the edge and with small area
    '''
    idx = remove_edge_idx(boxes,thickness,image_size)
    bboxes,cclasses,sscores = get_by_index(idx,boxes,classes,scores)

    return bboxes,cclasses,sscores

def select_classes(boxes,classes,scores,COI):
    '''Select classes of interst
    '''
    idx = select_COI_idx(classes,COI)
    bboxes,cclasses,sscores = get_by_index(idx,boxes,classes,scores)

    return bboxes,cclasses,sscores

def non_max_suppression(boxes,classes,scores,max_bbox_overlap):
    '''non_max_suppression with scores.
    '''
    idx = non_max_suppression_idx(boxes,max_bbox_overlap,scores)
    bboxes,cclasses,sscores = get_by_index(idx,boxes,classes,scores)

    return bboxes,cclasses,sscores
