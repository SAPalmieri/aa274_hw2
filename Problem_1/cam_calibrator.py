#!/usr/bin/python

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np
import glob

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]


    def genCornerCoordinates(self, u_meas, v_meas):
        '''
        Inputs:
            u_meas: a list of arrays where each array are the u values for each board.
            v_meas: a list of arrays where each array are the v values for each board.
        Output:
            corner_coordinates: a tuple (Xg, Yg) where Xg/Yg is a list of arrays where each array are the x/y values for each board.

        HINT: u_meas, v_meas starts at the blue end, and finishes with the pink end
        HINT: our solution does not use the u_meas and v_meas values 
        HINT: it does not matter where your frame it, as long as you are consistent! 
        '''

        ########## Code starts here ##########
        mylistx = []
        mylisty = []
        tempx = []
        tempy = []
        for iterboard in range(self.n_chessboards): # for each chessboard
            for j in range(self.n_corners_y): #for each row in x
                for i in range(self.n_corners_x): # for each corner in y
                    xval = i*self.d_square
                    yval = j*self.d_square
                    tempx.append(xval)
                    tempy.append(yval)
            
            mylistx.insert(iterboard,tempx)
            mylisty.insert(iterboard,tempy)
            tempx = []
            tempy = []
        corner_coordinates = ( np.array(mylistx), np.array(mylisty) )  # UPDATE ME

        ########## Code ends here ##########

        return corner_coordinates
    def estimateHomography(self, u_meas, v_meas, X, Y):    # Zhang Appendix A
        '''
        Inputs:
            u_meas: an array of the u values for a board.
            v_meas: an array of the v values for a board.
            X: an array of the X values for a board. (from genCornerCoordinates)
            Y: an array of the Y values for a board. (from genCornerCoordinates)
        Output:
            H: the homography matrix. its size is 3x3
        
        HINT: What is the size of the matrix L? - (2n x 9)
        HINT: What are the outputs of the np.linalg.svd function? Based on this, where does the eigenvector corresponding to the smallest eigen value live?
        HINT: np.stack and/or np.hstack may come in handy here.
        '''
        ########## Code starts here ##########
        X = np.reshape(X,(63,1))
        Y = np.reshape(Y,(63,1))
        zeroT = np.zeros((63,3))
        u_meas = np.reshape(u_meas,(63,1))
        v_meas = np.reshape(v_meas,(63,1))
        MtildeT = np.hstack( ( X,Y,np.ones(np.shape(X))) )

        L1 = np.hstack( (MtildeT, zeroT, np.multiply(-u_meas, MtildeT)) )
        L2 = np.hstack( (zeroT, MtildeT, np.multiply(-v_meas,MtildeT) ), )
       
        L = np.vstack((L1,L2))
        u, s, vh = np.linalg.svd(L)
        x = vh[8,:]
        H = np.vstack((x[0:3], x[3:6], x[6:])) # UPDATE ME
        #H = H.T
        ########## Code ends here ##########
        return H

    def getCameraIntrinsics(self, H):    # Zhang 3.1, Appendix B
        '''
        Input:
            H: a list of homography matrices for each board
        Output:
            A: the camera intrinsic matrix

        HINT: MAKE SURE YOU READ SECTION 3.1 THOROUGHLY!!! V. IMPORTANT
        HINT: What is the definition of h_ij?  
        HINT: It might be cleaner to write an inner function (a function inside the getCameraIntrinsics function)
        HINT: What is the size of V? - 2nx6 where n = 23 (images)
        '''
        ########## Code starts here ##########

        def getVij(h):
            '''
            inputs: h - estimate of homography
            outputs: v12,v11,v22  for each board
            '''
            h = h.T
            i = 0
            j = 1
            v12 = np.array([ 
                            [np.multiply(h[i,0],h[j,0]) ], 
                            [np.multiply(h[i,0],h[j,1]) + np.multiply(h[i,1],h[j,0]) ],
                            [np.multiply(h[i,1],h[j,1]) ], 
                            [np.multiply(h[i,2],h[j,0]) + np.multiply(h[i,0],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,1]) + np.multiply(h[i,1],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,2])]
                            ])
            i = 0
            j = 0
            v11 = np.array([ 
                            [np.multiply(h[i,0],h[j,0]) ], 
                            [np.multiply(h[i,0],h[j,1]) + np.multiply(h[i,1],h[j,0]) ],
                            [np.multiply(h[i,1],h[j,1]) ], 
                            [np.multiply(h[i,2],h[j,0]) + np.multiply(h[i,0],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,1]) + np.multiply(h[i,1],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,2])]
                            ])
            i = 1
            j = 1
            v22 = np.array([ 
                            [np.multiply(h[i,0],h[j,0]) ], 
                            [np.multiply(h[i,0],h[j,1]) + np.multiply(h[i,1],h[j,0]) ],
                            [np.multiply(h[i,1],h[j,1]) ], 
                            [np.multiply(h[i,2],h[j,0]) + np.multiply(h[i,0],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,1]) + np.multiply(h[i,1],h[j,2]) ],
                            [np.multiply(h[i,2],h[j,2])]
                            ])
            Vij = np.vstack((v12.T,v11.T-v22.T))
            Vij = np.reshape(Vij,(2,6))
            return Vij

        Vtot = []
        
        n = self.n_chessboards
        for image in range(self.n_chessboards):
            V = getVij(H[image])
            Vtot.append(V)
            
        Vtot = np.reshape(Vtot,(2*n,6))    
        u,s,vh = np.linalg.svd(Vtot)
        sv = vh[5,:]
        b = np.array([[sv[0], sv[1], sv[3]],
                    [sv[1], sv[2], sv[4]],
                    [sv[3], sv[4], sv[5]] ])
                    
        v_0 = (b[0,1]*b[0,2]-b[0,0]*b[1,2]) / (b[0,0]*b[1,1]-b[0,1]**2)
        lamb = b[2,2]- (b[0,2]**2 + v_0* (b[0,1]*b[0,2]-b[0,0]*b[1,2]))/b[0,0]
        alp = np.sqrt(lamb/b[0,0])
        bet = np.sqrt(lamb*b[0,0]/(b[0,0]*b[1,1]-b[0,1]**2))
        gam = -b[0,1]*(alp**2)*bet/lamb
        u_0 = gam*v_0/bet - b[0,2]*(alp**2)/lamb
        A = np.array([[alp, gam, u_0],
                    [0, bet, v_0],
                    [0,0,1] ]) # UPDATE ME
        
        ########## Code ends here ##########
        return A

    def getExtrinsics(self, H, A):    # Zhang 3.1, Appendix C
        '''
        Inputs:
            H: a single homography matrix
            A: the camera intrinsic matrix
        Outputs:
            R: the rotation matrix
            t: the translation vector
        '''
        ########## Code starts here ##########
        # UPDATE ME
        Ainv = np.linalg.inv(A)
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lamb = 1 /  np.linalg.norm(np.dot(Ainv,h1))
        r1comp = np.multiply(lamb,np.dot(Ainv,h1))
        r2comp = np.multiply(lamb,np.dot(Ainv,h2))
        r3comp = np.cross(r1comp,r2comp)
        t =  np.multiply(lamb,np.dot(Ainv,h3)) 
        t = np.reshape(t,(3,1))

        r1comp = np.reshape(r1comp, (3,1))    
        r2comp =  np.reshape(r2comp, (3,1))  
        r3comp =  np.reshape(r3comp, (3,1)) 
        Rtest = np.column_stack((r1comp,r2comp,r3comp))
        
        u,s,vh = np.linalg.svd(Rtest)
        R = np.dot(u,vh)
    
        ########## Code ends here ##########
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x, y: the coordinates in the ideal normalized image plane

        '''
        ########## Code starts here ##########
        # UPDATE ME
        Mtilde = np.row_stack((X,Y,Z, np.ones((np.shape(X)))))
        extrinsic = np.column_stack((R,t))
        
        smtilde = np.dot(extrinsic, Mtilde)
        x = smtilde[0]
        y = smtilde[1]

        ########## Code ends here ##########
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u, v: the coordinates in the ideal pixel image plane
        '''
        ########## Code starts here ##########
        # UPDATE ME
        Mtilde = np.row_stack((X,Y,Z, np.ones((np.shape(X)))))
        extrinsic = np.column_stack((R,t))
        smtilde = np.dot(A,np.dot(extrinsic,Mtilde))

        # scale = np.linalg.norm(smtilde[2])
        u = np.divide(smtilde[0], smtilde[2])
        v = np.divide(smtilde[1], smtilde[2])
        ########## Code ends here ##########
        return u, v

    def transformWorld2NormImageDist(self, X, Y, R, t, k):    # Zhang 3.3
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x_br, y_br: the real, normalized image coordinates         
        '''
        ########## Code starts here ##########        
                # UPDATE ME
        X,Y = CameraCalibrator.transformWorld2NormImageUndist(self, X, Y, Z, R, t)
        x_br = X + np.multiply(X, np.multiply(k[0], X**2 + Y**2)) + np.multiply(X, np.multiply(k[1], (X**2 + Y**2)**2 ))
        y_br = Y + np.multiply(Y, np.multiply(k[0], X**2 + Y**2)) + np.multiply(Y, np.multiply(k[1], (X**2 + Y**2)**2 ))

        ########## Code ends here ##########        
        return x_br, y_br

    def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):    # Zhang 3.3
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters                     
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u_br, v_br: the real, observed image coordinates         
        '''
        ########## Code starts here ##########                
        # UPDATE ME
        u0 = A[0,2]
        v0 = A[1,2]
        X,Y = CameraCalibrator.transformWorld2NormImageUndist(self, X, Y, Z, R, t)
        u,v = CameraCalibrator.transformWorld2PixImageUndist(self, X, Y, Z, R, t, A)
        
        u_br = u + np.multiply( (u-u0), np.multiply(k[0], X**2 + Y**2)) + np.multiply(u-u0, np.multiply(k[1],(X**2 + Y**2)**2  ))
        v_br = v + np.multiply( (v-v0), np.multiply(k[0], X**2 + Y**2)) + np.multiply(v-v0, np.multiply(k[1],(X**2 + Y**2)**2  ))
        ########## Code ends here ##########
        return u_br, v_br

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2


        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            if k[0] != 0:
                u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
                ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_10ubplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)

