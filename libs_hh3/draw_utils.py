"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def displayCamera(R, t, ax, cam_scale=1, Z_scale=3):
      """
      with the notation P = K [R, t]
                        C = -R * t
      """
      R = R.T;
      C = -np.dot(R, t);
      
      window11 = cam_scale * np.array([1,  1, Z_scale]).reshape(3, 1); 
      window12 = cam_scale * np.array([-1, 1, Z_scale]).reshape(3, 1);  
      window21 = cam_scale * np.array([-1,-1, Z_scale]).reshape(3, 1);     
      window22 = cam_scale * np.array([1, -1, Z_scale]).reshape(3, 1); 
      
      windowPrime11 = np.dot(R, window11) + C.reshape(3, 1); 
      windowPrime12 = np.dot(R, window12) + C.reshape(3, 1); 
      windowPrime21 = np.dot(R, window21) + C.reshape(3, 1); 
      windowPrime22 = np.dot(R, window22) + C.reshape(3, 1);
      
      ax_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5)]
      Z_scale = 1
      # axis
      for i in range(3):
            Xs = np.hstack((C[0], C[0]+cam_scale*R[0,i]))
            Ys = np.hstack((C[1], C[1]+cam_scale*R[1,i]))
            Zs = np.hstack((C[2], C[2]+cam_scale*R[2,i]))
            ax.plot(Xs, Ys, Zs, c = ax_colors[i])            
      col = ax_colors[-1]
      # cones
      ax.plot(np.hstack((windowPrime11[0], C[0])), np.hstack((windowPrime11[1], C[1])), np.hstack((windowPrime11[2], C[2])), c=col)
      ax.plot(np.hstack((windowPrime12[0], C[0])), np.hstack((windowPrime12[1], C[1])), np.hstack((windowPrime12[2], C[2])), c=col)
      ax.plot(np.hstack((windowPrime22[0], C[0])), np.hstack((windowPrime22[1], C[1])), np.hstack((windowPrime22[2], C[2])), c=col)
      ax.plot(np.hstack((windowPrime21[0], C[0])), np.hstack((windowPrime21[1], C[1])), np.hstack((windowPrime21[2], C[2])), c=col)
      # furthest plane
      Xs = np.hstack((windowPrime11[0], windowPrime12[0], windowPrime21[0], windowPrime22[0], windowPrime11[0]))
      Ys = np.hstack((windowPrime11[1], windowPrime12[1], windowPrime21[1], windowPrime22[1], windowPrime11[1]))
      Zs = np.hstack((windowPrime11[2], windowPrime12[2], windowPrime21[2], windowPrime22[2], windowPrime11[2]))
      ax.plot(Xs, Ys, Zs, c = col)
      return ax
      


def drawLines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    if len(img1.shape) == 2:
        r, c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    else:
        r, c, _ = img1.shape 
    # draw the lines  
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(map(int, pt1)),5,color,-1)
        img2 = cv2.circle(img2,tuple(map(int, pt2)),5,color,-1)
    return img1,img2

