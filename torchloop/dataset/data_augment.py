import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
test_img = "/media/bigtree/DATA/data_ubuntu/test_img.jpg"

#####
# color caliberation
#####

#####
# affine/perspective transformation
#####

######
# helper func: determine a img's w and h
######
def dim_img(img_path: str) -> None:
    assert os.path.exists(img_path)
    img_ = cv2.imread(img_path)
    h1, w1, c1 = img_.shape
    return h1, w1, c1

####
# first function show image
####
def show_img(img_path: str, win_name: str="img_win") -> None:
    assert os.path.exists(img_path)
    img_ = cv2.imread(img_path)
    cv2.imshow(win_name, img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

####
# scale img
####
def scale_img(img_path: str, scaled_w: int, scaled_h: int,
        win_name: str="img_win") -> None:
    assert os.path.exists(img_path)
    img_ = cv2.imread(img_path)
    res = cv2.resize(img_, (scaled_w, scaled_h), 
            interpolation=cv2.INTER_CUBIC)
    cv2.imshow(win_name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

######
# translate img
######
def translate_img(img_path: str, tx: int, ty: int,
        win_name: str="img_win") -> None:
    assert os.path.exists(img_path)
    # img_ = cv2.imread(img_path, 0) # 0 means grey scale
    img_ = cv2.imread(img_path) 

    # rows, cols = img_.shape # only work when it's grey scale
    rows, cols, __ = img_.shape 
    ####
    # translation matrix (100, 50)
    ####
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_ = cv2.warpAffine(img_, M, (cols,rows))

    cv2.imshow(win_name, img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#####
# rotate img
#####
def rotate_img(img_path: str, center_w: int, center_h: int,
        rotate_angle: int, win_name: str="img_win") -> None:
    assert os.path.exists(img_path)
    img_ = cv2.imread(img_path) # 0 means grey scale
    rows, cols, __ = img_.shape
    
    rotation_mat = cv2.getRotationMatrix2D((center_w, center_h), 
            rotate_angle, 1)
    dst = cv2.warpAffine(img_, rotation_mat, (cols,rows))

    cv2.imshow(win_name, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

######
# affine transformation
######
def affine_img(img_path: str) -> None:
    assert os.path.exists(img_path)
    img_ = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    #####
    # the below line required when use skimage based pyplot
    #####
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    rows, cols, __ = img_.shape

    #####
    # he wants to get pts1 transform 2 pts2
    #####
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    
    dst = cv2.warpAffine(img_, M, (cols,rows))
    
    plt.subplot(121),plt.imshow(img_),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

######
# perspective transformation
######
def perspective_img_naive(img_path: str) -> None:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows,cols,ch = img.shape
    
    pts1 = np.float32([[20, 20],[610, 20],[20, 200],[610, 200]])
    pts2 = np.float32([[0, 0], [600, 0], [0, 200], [600, 200]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
 
    dst = cv2.warpPerspective(img, M, (600, 200))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show() 

def perspective_img(img_path: str, 
        transit_from: List[List[int]],
        transit_to: List[List[int]]) -> None:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows,cols,ch = img.shape
    
    # pts1 = np.float32([[20, 20],[610, 20],[20, 200],[610, 200]])
    # pts2 = np.float32([[0, 0], [600, 0], [0, 200], [600, 200]])

    pts1 = np.float32(transit_from)
    pts2 = np.float32(transit_to)
 
    M = cv2.getPerspectiveTransform(pts1, pts2)
 
    dst = cv2.warpPerspective(img, M, (600, 200))
 
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show() 

#####
# gan based augmentation
#####


#####
# neural transfer based augmentation
#####
