# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:42:16 2018
Offline Signature Verification 

@author: Vishal Jeswani Kirti Mahajan Swapnil Birajdar Harish GV
"""
#Importing Libraries 
from pylab import *
import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
from sklearn import svm
import imagehash
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import linear_model
import csv


#Importing Images from Folders Genuine : Genuine , Forge for Uploading the testing images
genuine_image_filenames = listdir("C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\real")
forged_image_filenames = listdir("C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\test")

genuine_image_paths = "C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\real"
forged_image_paths = "C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\test"

genuine_image_features = [[] for x in range(100)]
forged_image_features = [[] for x in range(100)]


for name in genuine_image_filenames:
    signature_id = int(name[:3])
    genuine_image_features[signature_id - 1].append({"name": name})

a1=0
for a1 in range(len(forged_image_filenames)):
    name1=forged_image_filenames[a1]
    signature_id1 = int(name1[5:8])
    forged_image_features[a1].append({"name": name1})
   


#Creating function to Preprocess the image
def preprocess_image(path, display=False):
    raw_image = cv2.imread(path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image

    if display:
        cv2.imshow("RGB to Gray", bw_image)
        cv2.waitKey()

    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)

    if display:
        cv2.imshow("Threshold", threshold_image)
        cv2.waitKey()

    return threshold_image

#Creating Function to get contours
def get_contour_features(im, display=False):
    '''
    :param im: input preprocessed image
    :param display: flag - if true display images
    :return:aspect ratio of bounding rectangle, area of : bounding rectangle, contours and convex hull
    '''

    rect = cv2.minAreaRect(cv2.findNonZero(im))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(im.copy(), [box], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    hull = cv2.convexHull(cv2.findNonZero(im))

    if display:
        convex_hull_image = cv2.drawContours(im.copy(), [hull], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(convex_hull_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    im2, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = cv2.drawContours(im.copy(), contours, -1, (120, 120, 120), 3)
        cv2.imshow("a", cv2.resize(contour_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area

des_list = []

#Creating Function to get SIFT features
def sift(im, path, display=False):
    raw_image = cv2.imread(path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(im, None)

    if display:
        cv2.drawKeypoints(im, kp, raw_image)
        cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
        cv2.waitKey()

    return (path, des)

def cont1(path,display=False):
    og = cv2.imread(path)
    og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    ret, temp_thre = cv2.threshold(og_gray, 200, 255,10)
    _, contours1, hierarchy = cv2.findContours(temp_thre,1,2)

    temp_cont = contours1[0]
    return (temp_cont)


#Processing Starts here
cor = 0
wrong = 0

im_contour_features = []
ch_contour_features = []
tot_train_genuine = np.empty((0,54), float32)
tot_test_genuine = np.empty((0,54), float32)
tot_train_forge = np.empty((0,54), float32)
tot_test_forge = np.empty((0,54), float32)

for i in range(signature_id):
    des_list = []
    for im in genuine_image_features[i]:
        image_path = genuine_image_paths + "/" + im['name']
        preprocessed_image = preprocess_image(image_path)
        hash = imagehash.phash(Image.open(image_path))

        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            get_contour_features(preprocessed_image.copy(), display=False)

        hash = int(str(hash), 16)
        im['hash'] = hash
        im['aspect_ratio'] = aspect_ratio
        im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
        im['contour_area/bounding_area'] = contours_area / bounding_rect_area
        
        in_kp1, in_des1 = sift(preprocessed_image,image_path)
        temp_cont1=cont1(image_path)
        #og = cv2.imread(image_path)
        #og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
        #ret, temp_thre = cv2.threshold(og_gray, 200, 255,10)
        #_, contours1, hierarchy = cv2.findContours(temp_thre,1,2)
        
        #temp_cont = contours1[0]
        #return (path)

        
        im['temp_cont1']=temp_cont1
        im['in_kp1']=in_kp1
        im['in_des1']=in_des1
        im_contour_features.append([hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

        des_list.append(sift(preprocessed_image, image_path))

ii=0
for ii in range(len(forged_image_filenames)):

    for ch in forged_image_features[ii]:
        image_path = forged_image_paths + "/" + ch['name']
        preprocessed_image = preprocess_image(image_path)
        hash = imagehash.phash(Image.open(image_path))

        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            get_contour_features(preprocessed_image.copy(), display=False)

        hash = int(str(hash), 16)
        ch['hash'] = hash
        ch['aspect_ratio'] = aspect_ratio
        ch['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
        ch['contour_area/bounding_area'] = contours_area / bounding_rect_area
        
        temp_cont2=cont1(image_path)
        in_kp2, in_des2 = sift(preprocessed_image,image_path)
        ch['temp_cont1']=temp_cont1
        ch['in_kp1']=in_kp1
        ch['in_des1']=in_des1
        ch_contour_features.append([hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

        des_list.append(sift(preprocessed_image, image_path))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    k = 50
    voc, variance = kmeans(descriptors, k, 1)

    final_features=im_contour_features+ch_contour_features
    # Calculate the histogram of features
    
    vi=0
    im_features = np.zeros((len(genuine_image_features[i]) + len(forged_image_features[i]), k+4), "float32")
    for vi in range(len(genuine_image_features[vi]) + len(forged_image_features[vi])):
        words, distance = vq(des_list[vi][1], voc)
        for w in words:
            im_features[vi][w] += 1

        for j in range(4):
            im_features[vi][k+j] = final_features[vi][j]
    
final_output=[]

#Comparing each test image with the respective user genuine images
x=0
for x in range(len(forged_image_filenames)):
    test_features=forged_image_features[x][0]
    test_name=test_features.get("name")
    test_user_id = int(test_name[5:8])
    test_in_des1=test_features.get("in_des1")
    test_cont1=test_features.get("temp_cont1")
    
    check_features=genuine_image_features[test_user_id-1]
    y=0
    sift_corr=0
    sift_wrong=0
    cont_right=0
    cont_wrong=0
    for y in range(len(check_features)):
        check_features1=check_features[y]
        check_features_name=check_features1.get("name")
        check_in_des1=check_features1.get("in_des1")
        check_cont1=check_features1.get("temp_cont1")
        
        
        gray1= preprocess_image(genuine_image_paths + "/" + check_features_name)
        gray2= preprocess_image(forged_image_paths + "/" + test_name)

        
        sift1 = cv2.xfeatures2d.SIFT_create()
        check_kp1, check_in_des1 = sift1.detectAndCompute(gray1,None)
        sift2 = cv2.xfeatures2d.SIFT_create()
        tes_kp2, test_in_des1 = sift2.detectAndCompute(gray2,None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(test_in_des1,check_in_des1, k=2)
        # Apply ratio test
        
        good = []
        for m,n in matches:
            if m.distance < 0.98*n.distance:
                good.append([m])
                a=len(good)
                percent=(a*100)/max(len(test_in_des1),len(check_in_des1))
        if percent >= 50.00:
            sift_corr=sift_corr+1
        if percent < 50.00:
            sift_wrong=sift_wrong+1
        
        og = cv2.imread(genuine_image_paths + "/" + check_features_name)
        og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
        ret, temp_thre = cv2.threshold(og_gray, 200, 255,10)
        _, contours, hierarchy = cv2.findContours(temp_thre,1,2)
        
        temp_cont1 = contours[0]
        
        dup = cv2.imread(forged_image_paths + "/" + test_name)
        dup_gray = cv2.cvtColor(dup, cv2.COLOR_BGR2GRAY)
        ret, tar_thr = cv2.threshold(dup_gray, 200, 255, 10)
        
        
        _, contours, hierarchy = cv2.findContours(tar_thr,1,2)
        
        temp_cont2 = contours[0]
        
        for c in contours:
            match = cv2.matchShapes(temp_cont1, temp_cont2,1 , 0)
          
            if match <= 0.2:
                cont_right=cont_right+1
            else:
                cont_wrong=cont_wrong+1

        
    sift_final=sift_corr/(sift_wrong+sift_corr)
    if sift_final > 0.49 :
        sift_status="Im_Match"
    if sift_final <= 0.49 :
        sift_status="Im_Not_Match"
    
    cont_final=cont_right/(cont_wrong+cont_right)
    if cont_final > 0.49 :
        cont_status="Genuine"
    if cont_final <= 0.49 :
        cont_status="Forged"

    final_output.append([test_name,test_user_id,sift_status,cont_status])

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(final_output)   
