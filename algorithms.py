import numpy as np 
import cv2 as cv 
import glob
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




"""
Dans cette calsse on va implementer les Algorithmes pour
    -> calibrer la caméra
    -> estimer les points 3D à partir d'une paire d'images stereo(Simple stereo System)

la classe Algorithm est importée as algo pour faire appel à ces méthodes dans l'interface graphique
de la bibliothéque dearpygui.
"""

#TODO: Géneraliser le processus de calibration avec différents patterns
# cube, échiquier de différents dimensions ...etc

class Algorithm: 

    
    # @param folder_path : chemin vers le fichier ou se trouve les images
    # de l'echequier pour calibrer la camera (soient entre 20 -30 images pour avoir de bon résultats)
    # 
    # @param return mtx : la matrice de calibration (matrice des parametres intrinsincs)
    def calibrer_camera(folder_path): 
        # Defining the dimensions of checkerboard      
        CHECKERBOARD = (9,7)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  
        # max number of iterations=30

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 
        error =[]

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        nx = 7
        ny = 9      #(7,9)

        # Extracting path of individual image stored in a given directory
        path = folder_path + '/*.png'
        images = glob.glob(path)

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)
            #ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
            
            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            

        cv.destroyAllWindows()

        h,w = img.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        rvecs_list = [np.array(r).tolist() for r in rvecs]
        tvecs_list = [np.array(t).tolist() for t in tvecs]

        # for debugging purposes
        def print_calib_matrix(): 
            print(f"Matrce de calibration : \n{mtx}")

        return mtx


    # stereo system : estimer la profondeur d'ou la 3D
    # 
    # @param mtx : la matrice des parametres intrinsincs de la caméra
    # @param disparity : la translation horizontale de la camera dans le monde réel
    # @params imageLpath, imageR_path : les images gauche et droite apres translation respectivement
    # 
    # @param return matched_image: l'image avec les points SIFT appareillés
    # @param return X,Y,Z: les coordonnées 3D de ces points SIFT
    def calculer_pts3D(mtx, disparity, imageL_path, imageR_path):

        sift = cv.SIFT_create()

        img1 = cv.imread(imageL_path,0) # queryImage
        img2 = cv.imread(imageR_path,0) # trainImage
   
       
        
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])      

        img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        
        
        # Les coordonnes du plan image (ul,vl),(ur,vr)
        coords = []
        for match in good: 
            kp1_idx = match[0].queryIdx
            kp2_idx = match[0].trainIdx
            kpp1 = kp1[kp1_idx]
            kpp2 = kp2[kp2_idx]

            # Prendre les coordonner du point en commun
            ul, vl = kpp1.pt
            ur, vr = kpp2.pt

            coords.append([(ul,vl),(ur,vr)])

        #Caucul de xc,yc et zc en utilisant fx, fy, ox, oy
        Ox = mtx[0][2]
        Oy = mtx[1][2]
        fx = mtx[0][0]
        fy = mtx[1][1]
        b  = disparity 

        camera_coords = []
        for coord in coords: 
            xc =(b*(coord[0][0]-Ox))/(coord[0][0]-coord[1][0])
            # xc=(b*(ul-Ox))/(ul-ur)

            yc =(b*fx*(coord[0][0]-Ox))/(fy*(coord[0][0]-coord[1][0]))
            # yc=(b*fx*(ul-Ox))/(fy*(ul-ur))

            zc =(b*fx)/(coord[0][0]-coord[1][0])
            # zc=(b*fx)/(ul-ur) 

            camera_coords.append((xc,yc,zc))
        
        
        for i in range (len(camera_coords)):
            print(f"point {i} has coordinates : {camera_coords[i][0] },{camera_coords[i][1] },{camera_coords[i][2] }")

        return img_matches, camera_coords

        


# Cette fonction sert à tester et debugger
def test():
    
    # _______________ test de la calibration
    folder_path = r"C:\Users\monms\Desktop\MIV\VISION\TP-04\myImages"
    calibration_matrix = Algorithm.calibrer_camera(folder_path)
    print(f"Matrice de calibration : \n {calibration_matrix}")


    #_________________ test de depth estimation
    calibration_matrix =[[8.54832131e+03, 0.00000000e+00, 9.41016774e+02],
                        [0.00000000e+00, 5.47223383e+04, 1.26355037e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


    folder_path = r"C:\Users\monms\Desktop\MIV\VISION\TP-04\myImages"
    # calibration_matrix = Algorithm.calibrer_camera(folder_path)
    imageL_path =   r"C:\Users\monms\Desktop\imageL.jpg"
    imageR_path =   r"C:\Users\monms\Desktop\imageR.jpg"


    image, _3d_points =  Algorithm.calculer_pts3D(calibration_matrix, 20.0, imageL_path, imageR_path )

    image, _3d_points =  Algorithm.calculer_pts3D(calibration_matrix, 20.0, imageL_path, imageR_path )


    points_arr = np.array(_3d_points)
    points_arr[:, 2] /= 10
    # Create an Open3D point cloud from the numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)

    # Save the point cloud in PLY format
    o3d.io.write_point_cloud("point_cloud.ply", pcd)

# test()