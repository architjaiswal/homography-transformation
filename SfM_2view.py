"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Sample test 
# impath_left = 'data/uf_left.png'
# impath_right = 'data/uf_right.png'
# K = np.loadtxt('data/K_iphone_reduced.txt')
# dist = np.zeros(5) # use the actual lens distortions params if available

## Use your data
impath_left = 'data/image_left.png'
impath_right = 'data/image_right.png'
K = np.loadtxt('data/K_piCam.txt')
dist = np.loadtxt('data/dist_piCam.txt')

# our library
from libs_hh3.geo3D import SceneReconstruction3D
recon_3D = SceneReconstruction3D(K, dist)

# the SFM pipeline
recon_3D.load_image_pair(impath_left, impath_right)
recon_3D._extract_keypoints_sift()
recon_3D._estimate_fundamental_matrix()
recon_3D.draw_epipolar_lines()
recon_3D._estimate_essential_matrix()
recon_3D._find_camera_matrices_rt()
recon_3D._find_projection_matrices()
recon_3D._triangulate_3d_points()
recon_3D.plot_point_cloud()
recon_3D._triangulate_and_plot_3d_points()
