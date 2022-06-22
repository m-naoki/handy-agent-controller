import os
import cv2
import mediapipe as mp
import time
from multiprocessing import Process, Manager, Queue
from handy_rl_controller.utils.joint_handling import get_joint_positions, Visualizer3D
import numpy as np
import yaml

class HandPoseDetector:
    def __init__(self, render=False):
        self.render = render
        file_path = os.path.dirname(__file__)
        relative_path = os.path.join(file_path, './cfg/mediapipe_info.yaml')
        with open(relative_path, 'r') as yml:
            self.cfg = yaml.safe_load(yml)
        
        self.q = Queue(10)
        # detector is executed as other process
        p = Process(target=self.estimate_hand_pose, args=(self.q,))
        p.start()
        
    def estimate_hand_pose(self, q):
        cap = cv2.VideoCapture(-1)
        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        ptime = 0
        
        while True:
            ret, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    # Drawing gotten landmarks in the image
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        if id in [4, 8, 12, 16, 20]:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            
            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
            
            item = {"multi_hand_landmarks":results.multi_hand_landmarks,
                        "fps": fps}
            if not q.full():
                q.put(item, block=True)
            if self.render:
                cv2.putText(img, 'FPS:{:.1f}'.format(fps), (18,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
                cv2.imshow("img", img)
                cv2.waitKey(1)

    def __take_landmarks_rawdata(self):
        result = None
        # block process until get result
        while result is None:
            # get latest result and clear Queue
            while not self.q.empty():
                result = self.q.get(block=True)
                # check the result containes detected hand
                if result['multi_hand_landmarks'] is None:
                    result = None
        # post processing
        return result['multi_hand_landmarks'][0]
    
    def take_landmarks(self):
        landmarks = self.__take_landmarks_rawdata()
        landmarks_dict = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            l = {}
            for joint in ['first', 'second', 'third', 'fourth']:
                l[joint] = landmarks.landmark[self.cfg['mediapipe'][finger][joint]['offset']]
            # add wrist coord at last
            l['wrist'] = landmarks.landmark[self.cfg['mediapipe']['wrist']['offset']]
            landmarks_dict[finger] = l
        return landmarks_dict
    
    def take_joint_angles(self):
        landmarks_dict = self.take_landmarks()
        angles = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            a = {}
            for j1, j2, j3 in [['first', 'second', 'third'],
                                ['second', 'third', 'fourth'],
                                ['third', 'fourth', 'wrist']]:
                vec1 = [landmarks_dict[finger][j1].x - landmarks_dict[finger][j2].x,
                        landmarks_dict[finger][j1].y - landmarks_dict[finger][j2].y,
                        landmarks_dict[finger][j1].z - landmarks_dict[finger][j2].z
                        ]
                unit_vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = [landmarks_dict[finger][j2].x - landmarks_dict[finger][j3].x,
                        landmarks_dict[finger][j2].y - landmarks_dict[finger][j3].y,
                        landmarks_dict[finger][j2].z - landmarks_dict[finger][j3].z
                        ]
                unit_vec2 = vec2 / np.linalg.norm(vec2)
                rad = np.arccos(np.dot(unit_vec1, unit_vec2))
                # rad is 0.0 ~ 3.141516
                a[j1] = rad
            angles[finger] = a
        return angles


if __name__ == "__main__":
    hpd = HandPoseDetector(True)
    vis = Visualizer3D()
    
    # 3DAxesを追加
    while True:
        angles = hpd.take_joint_angles()