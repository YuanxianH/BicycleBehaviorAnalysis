import cv2
import numpy as np

def find_correspondence_points(img1, img2, feature='SURF',limit = 9999):
    '''Find Correspondence Points
    Parameters:
    ==========
    img1,img2 -> two similar images
    feature -> choose the feature used(SIFT or SURF)
    limit -> the limit number of feature points

    Return:
    ======
    pts1,pts2 -> array(2 x m) saved catesian coordination
    '''
    assert (feature=='SURF' or feature=='SIFT'),'Only support SURF and SIFT feature'
    if feature == 'SIFT':
        fea = cv2.xfeatures2d.SIFT_create(limit)
    elif feature == 'SURF':
        fea = cv2.xfeatures2d.SURF_create(limit)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = fea.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = fea.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T

def get_patch_by_box(img,box):
    '''get a patch in an image by the bouding box
    parameters:
    ==========
    img1: image in array format
    box: bouding box(l,t,w,h)

    returns:
    =======
    template: an array
    '''
    l,t,w,h = [int(i) for i in box]
    r = l + w;b = t + h
    t = max(0,t); l = max(0,l)
    b = min(img.shape[0],b); r = min(img.shape[1],r)
    template = img[t:b,l:r,:]
    return template

def match_images(img1,img2,box,offset=(-400,-50,0,50),method=cv2.TM_CCOEFF_NORMED):
    '''matching template around the box in another image
    parameters:
    ==========
    img1: image has template
    img2: searching in img2
    box: bouding box(x,y,w,h)
    offset:(l,t,r,b)

    returns:
    =======
    res: the result of matching template
    search_patch: the search range in the image
    box_matched: (left,top,right,bottom),the coordination of the searched patch
    '''

    l,t,w,h = box
    r = l + w;b = t + h
    template = get_patch_by_box(img1,box)
    # print("t,b,l,r",(t,b,l,r))

    # strict searching range
    ls,ts,rs,bs = np.array([l,t,r,b],dtype='int32') + \
                            np.array(offset,dtype='int32')
    # print("ts,bs,rs,ls",ts,bs,rs,ls)
    # over screen
    ts = max(0,ts);ls = max(0,ls)
    bs = min(img2.shape[0],bs); rs = min(img2.shape[1],rs)
    # print("ts,bs,ls,rs",(ts,bs,ls,rs))

    search_patch = img2[ts:bs,ls:rs,:]#search around the template

    res = cv2.matchTemplate(search_patch,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    # the matched patch's location
    l_matched = max_loc[0]+ls
    t_matched = max_loc[1]+ts
    r_matched = min(l_matched+w,img2.shape[1]-1)
    b_matched = min(t_matched+h,img2.shape[0]-1)

    box_matched = (l_matched,t_matched,r_matched,b_matched)
    return res,search_patch,box_matched
