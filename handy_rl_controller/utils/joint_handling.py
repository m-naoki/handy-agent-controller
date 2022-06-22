# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
# Image based imports
import cv2

def get_joint_positions(hand_landmarks):
    # Getting the wrist joint position in X and Y pixels 
    wrist_position = [
        hand_landmarks.landmark[0].x, 
        hand_landmarks.landmark[0].y, 
        hand_landmarks.landmark[0].z
        ] 

    # Getting the knuckle positions in X and Y pixels
    knucle_positions = {}
    
    for key, id in zip(['thumb', 'index', 'middle', 'ring', 'pinky'],
                        [1, 5, 9, 13, 17]):
        knucle_positions[key] = [
            hand_landmarks.landmark[id].x,
            hand_landmarks.landmark[id].y,
            hand_landmarks.landmark[id].z    
        ]
    
    finger_tip_positions = {}
    
    for key, id in zip(['thumb', 'index', 'middle', 'ring', 'pinky'],
                        [4, 8, 12, 16, 20]):
        finger_tip_positions[key] = [
            hand_landmarks.landmark[id].x,
            hand_landmarks.landmark[id].y,
            hand_landmarks.landmark[id].z    
        ]
    
    line_ids = {'thumb':[0,1,2,3,4],
                'index':[0,5,6,7,8],
                'middle':[0,9,10,11,12],
                'ring':[0,13,14,15,16],
                'pinky':[0,17,18,19,20]}
    finger_lines = {}
    for finger, ids in line_ids.items():
        finger_lines[finger] = {'x':[], 'y':[], 'z':[]}
        for id in ids:
            finger_lines[finger]['x'].append(hand_landmarks.landmark[id].x)
            finger_lines[finger]['y'].append(hand_landmarks.landmark[id].y)
            finger_lines[finger]['z'].append(hand_landmarks.landmark[id].z)

    return wrist_position, knucle_positions, finger_tip_positions, finger_lines
   
def check_hand_position(wrist_position, pinky_knuckle_position, wrist_joint_bound, pinky_knuckle_bound):
    # Checking if the pinky knuckle inside the pinky knuckle contour
    if cv2.pointPolygonTest(np.array(pinky_knuckle_bound), pinky_knuckle_position, False) > -1:
        # Checking if the wrist position is within the circular bound
             
        if wrist_position[0] <= (wrist_joint_bound.center[0] + wrist_joint_bound.radius) and wrist_position[0] >= (wrist_joint_bound.center[0] - wrist_joint_bound.radius) and wrist_position[1] > (wrist_joint_bound.center[1] - wrist_joint_bound.radius):
            return True
        else:
            print('check wrist!')
    else:
        print('check pinky!')
    return False

def get_approx_index(hand_landmarks):
    l0 = hand_landmarks.landmark[0]
    wrist_base = np.array((l0.x,l0.y,l0.z))

    l5 = hand_landmarks.landmark[5]
    index_base = np.array((l5.x,l5.y,l5.z))

    l6 = hand_landmarks.landmark[6]
    index_knuckle = np.array((l6.x,l6.y,l6.z))

    l7 = hand_landmarks.landmark[7]
    index_distal= np.array((l7.x,l7.y,l7.z))

    v0 = index_base - wrist_base
    v1 = index_knuckle - index_base
    v2 = index_distal - index_knuckle

    base_angle = np.arccos(np.dot(v0, v1 )/ (np.linalg.norm(v0) * np.linalg.norm(v1)))
    base_angle = np.clip(base_angle,0,1.6)

    index_angle = np.arccos(np.dot(v1, v2 )/ (np.linalg.norm(v1) * np.linalg.norm(v2)))
    index_angle = np.clip(index_angle,0,1.6)
    print (index_angle)
    return base_angle, index_angle
    
def get_approx_middle(hand_landmarks):
    l0 = hand_landmarks.landmark[0]
    wrist_base = np.array((l0.x,l0.y,l0.z))

    l9 = hand_landmarks.landmark[9]
    middle_base = np.array((l9.x,l9.y,l9.z))

    l10 = hand_landmarks.landmark[10]
    mid_kn = np.array((l10.x,l10.y,l10.z))

    l11 = hand_landmarks.landmark[11]
    mid_dist= np.array((l11.x,l11.y,l11.z))

    v0 = middle_base - wrist_base
    v1 = mid_kn - middle_base
    v2 = mid_dist - mid_kn

    base_mid_angle = np.arccos(np.dot(v0, v1 )/ (np.linalg.norm(v0) * np.linalg.norm(v1)))
    base_mid_angle = np.clip(base_mid_angle,0,1.5)

    mid_angle = np.arccos(np.dot(v1, v2 )/ (np.linalg.norm(v1) * np.linalg.norm(v2)))
    mid_angle = np.clip(mid_angle,0,1.5)
    print (mid_angle)
    return base_mid_angle, mid_angle


class Visualizer3D:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        
    def render(self, finger_lines):
        for key, line in finger_lines.items():
            self.ax.plot(line['x'], line['y'], line['z'])
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_zlim(0.0, 1.0)
        plt.draw()
        plt.pause(0.01)
        self.ax.cla()