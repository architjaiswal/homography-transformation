"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""
import cv2
import sys
import numpy as np
from os.path import isfile, join
from matplotlib import pyplot as plt
from libs_hh3.draw_utils import drawLines, displayCamera
from libs_hh3.triangulation import linear_LS_triangulation
from libs_hh3.triangulation import linear_eigen_triangulation


class SceneReconstruction3D:
    """3D scene reconstruction
        This class implements an algorithm for 3D scene reconstruction using
        structure-from-motion techniques.
    """
    def __init__(self, K, dist):
        """Constructor
            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist


    def load_image_pair(self, img_path1, img_path2):
        """Loads pair of image
            :param img_path1: path to first image
            :param img_path2: path to second image
        """
        self.img1 = cv2.imread(img_path1, 0)
        self.img2 = cv2.imread(img_path2, 0)
        # make sure images are valid
        if self.img1 is None or self.img2 is None:
            sys.exit("\n Image could not be loaded, check the paths. \n")


    def _extract_keypoints_sift(self):
        """extract keypoints and descriptors from both images"""
        ## cv2.xfeatures2d.SIFT_create() #for older versions
        sift = cv2.SIFT_create() 
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)
        img1 = cv2.drawKeypoints(self.img1, kp1, None)
        img2 = cv2.drawKeypoints(self.img2, kp2, None)
        # cv2.imshow('SIFT keypoints: left image', img1)
        # cv2.imshow('SIFT keypoints: right image', img2)
        plt.imshow(img1)
        plt.title('SIFT keypoints: left image')
        plt.show()
        plt.savefig('data/sift_keypoints_left.png')

        plt.imshow(img2)
        plt.title('SIFT keypoints: right image')
        plt.show()
        plt.savefig('data/sift_keypoints_right.png')

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('data/sift_keypoints_left.png', img1)
        # cv2.imwrite('data/sift_keypoints_right.png', img2)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        # knn search and ration test
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        self.match_pts1, self.match_pts2 = self._filterMatches_ratio_test(matches, kp1, kp2)


    def _filterMatches_ratio_test(self, matches, kp1, kp2):
        """ Filters sift feature matches based on Ratio test """
        good = []
        pts1, pts2 = [], []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)    
        pts1, pts2 = np.int32(pts1), np.int32(pts2)
        return pts1, pts2



    def draw_epipolar_lines(self):
        """Draws epipolar lines
            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("openpose")
        """
        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2re = self.match_pts2.reshape(-1, 1, 2)
        lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.F)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = drawLines(self.img1, self.img2, lines1, self.match_pts1, self.match_pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1.reshape(-1, 1, 2)
        lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = drawLines(self.img2, self.img1, lines2, self.match_pts2, self.match_pts1)
        # now show the drawing
        plt.subplot(121),plt.imshow(img1)
        plt.subplot(122),plt.imshow(img3)
        plt.savefig('data/epilines.png')
        plt.show()


    def _estimate_fundamental_matrix(self, do_ransac=False):
        """Estimates fundamental matrix """
        # your code (should be 1-2 lines)
        # you can use the cv2.findFundamentalMat function
        # if you want to use RANSAC, set do_ransac=True
        if do_ransac:
            self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1, self.match_pts2, cv2.FM_RANSAC)
        else:
            #self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1, self.match_pts2, cv2.FM_8POINT)
            self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1, self.match_pts2, cv2.FM_LMEDS)

        print("F = ", self.F)

    def _estimate_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        # your code (should be a one-liner)
        self.E = np.dot(np.dot(self.K.T, self.F), self.K)
        print("E = ", self.E)


    def _find_camera_matrices_rt(self):
        """Once essential matrix computation is done, this function
           finds the right [R|t] camera matrix out of all possible 4 configurations
           """
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        # Ensure rotation matrix are right-handed with positive determinant
        if np.linalg.det(np.dot(U, Vt)) < 0:
            Vt = -Vt
            
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,1.0]).reshape(3, 3)
        # iterate over all point correspondences used in the estimation of the F
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],  self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0], self.match_pts2[i][1], 1.0]))        

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        # first camera is the origin
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # Determine the correct choice of second camera matrix
        # only in one of the four configurations, all the points be in front of both cameras
        # Choices (See Hartley Zisserman 9.19)
        R1 = U.dot(W).dot(Vt)
        R2 = U.dot(W.T).dot(Vt)
        T1 = U[:, 2]
        T2 = - U[:, 2]
        self.R_all = [R1, R1, R2, R2]
        self.T_all = [T1, T2, T1, T2]
        # select the {R,t} for which the max points are in front of camera
        
        # your code to find the correct R, t
        # hint: make use the self._in_front_of_both_cameras function
        max_in_front = 0
        for i in range(4):
            R = self.R_all[i]
            T = self.T_all[i]
            in_front = self._in_front_of_both_cameras(R, T)
            if in_front > max_in_front:
                max_in_front = in_front
                self.Rt2 = np.hstack((R, T.reshape(3, 1)))

        self.Rt2 = np.hstack((R, T.reshape(3, 1)))
        
        

    def _find_projection_matrices(self):
        """Finds projection matrices from [R|t] matrices"""
        # your code: compute self.P1 and self.P2
        self.P1 = np.dot(self.K, self.Rt1)
        self.P2 = np.dot(self.K, self.Rt2)
        print("Left Projection Matrix P1 = ", self.P1)
        print("Right Projection Matrix P2 = ", self.P2)

          

    def _triangulate_3d_points(self):
        """Plots 3D point cloud
        """
        # reshape inlier points
        first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
        second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2]        
        # triangulate
        self.pts3D, _ = linear_eigen_triangulation(first_inliers, self.Rt1, second_inliers, self.Rt2)


    def _in_front_of_both_cameras(self, rot, trans):
        """Determines whether point correspondences are in front of both
           cameras
        """
        # get inliers for triangulation
        first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
        second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2] 
        Rt2 = np.hstack((rot, trans.reshape(3, 1)))
        pts3D, _ = linear_LS_triangulation(first_inliers, self.Rt1, second_inliers, Rt2)
        Npoints = self._check_Cheirality(pts3D, rot, trans)
        return Npoints
        
        
    def _check_Cheirality(self, X, rot, trans):
        """Determines how many triangulated points satisfy Cheirality conditions
        """
        R1, C1 = np.eye(3), np.zeros((1, 3))
        R2, C2 = rot.T, -np.dot(rot.T, trans)
        Npoints = 0
        if X.shape[0] > 0:
            # if point x is in front of camera {C, R}, condition is r3'(x-C) >0
            cam1 = np.dot((X-C1), R1[:,2]) > 0
            cam2 = np.dot((X-C2), R2[:,2]) > 0
            Npoints = sum(cam1*cam2 > 0) # true for both cameras    
        return Npoints


    def plot_point_cloud(self):
        """Plots 3D point cloud
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = displayCamera(self.Rt1[:, 0:3], self.Rt1[:, 3], ax, cam_scale=1)
        ax = displayCamera(self.Rt2[:, 0:3], self.Rt2[:, 3], ax, cam_scale=1)
        ax.plot(self.pts3D[:, 0], self.pts3D[:, 1], self.pts3D[:, 2], 'r.')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.savefig('data/3dps.png')
        plt.show()


    def _triangulate_and_plot_3d_points(self):
        """Plots 3D point cloud
        """
        # reshape inlier points
        first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
        second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2]
        # triangulate
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax = displayCamera(self.Rt1[:, 0:3], self.Rt1[:, 3], ax, cam_scale=1)
        # ax = displayCamera(self.Rt2[:, 0:3], self.Rt2[:, 3], ax, cam_scale=1)

        for i in range(4):
            R = self.R_all[i]
            T = self.T_all[i]
            Rt2 = np.hstack((R, T.reshape(3, 1)))
            self.pts3D, _ = linear_LS_triangulation(first_inliers, self.Rt1, second_inliers, Rt2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax = displayCamera(self.Rt1[:, 0:3], self.Rt1[:, 3], ax, cam_scale=1)
            ax = displayCamera(Rt2[:, 0:3], Rt2[:, 3], ax, cam_scale=1)
            ax.plot(self.pts3D[:, 0], self.pts3D[:, 1], self.pts3D[:, 2], 'r.')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            filename = 'data/3dps' + str(i) + '.png'
            plt.savefig(filename)
            plt.show()

