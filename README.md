# Signature-Verification-and-Fraud-Detection-using-OpenCV
Offline Signature Verification and Fraud Detection System using OpenCV

Input File structure

Files to be placed in 2 folders

Folder 1: Contains genuine images of all the users
Folder 2: Contains the images to be tested

Fileformat : xxxyyzzz.png
xxx : Signing user
yy: Image Id
zzz : Signature of the user

Output Format

Col 1 : Image Name
Col 2 : User ID
Col 3 : Checking if images are matched properly (ie both images of the same user)
Col 4 : Checking if images are forged image or genuine images (Main Requirement)

Methodologies used

1)Preproceesing
2)Feature extraction
  a)Contourts, Aspect Ratio , Hull Area , Boundry Area 
  b)SIFT - with and without clusters

3)Creating a classifier to identify features indicating forgery
  a)Logistic : Accuracy (0.44)
  b)Random Forest : Accuracy (0.52)
  c)SVM : Accuracy (0.46)
  
  However none of the models were indicative enough

4)Matching images based on 2 parameter features
  a)SIFT - Brute force Matching with KNN
  b)Contours - Shape Matching

Final Output : Giving 2 scores 
Image Matching : If both the images are of the same user
FraudDetection : If images are fraud or genuine 

Accuracy obtained : 83% (Average of 13 Samples)
