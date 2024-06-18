import math

import cv2
import mediapipe as mp
from playsound import playsound
import time
import pose_module as pm

detector = pm.PoseDetector()
landmarks = []
p_time = 0


def calculate_angle(origin, mid, end):
    x1, y1 = origin
    x2, y2 = mid
    x3, y3 = end

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def is_between(target, left, right):
    target_x, target_y = target
    lx, ly = left
    rx, ry = right

    return rx < target_x < lx


def find_mj():
    global landmarks
    global detector

    if landmarks:
        # Calculate angle between left shoulder, elbow and wrist to looking for straight left arm
        left_arm_angle = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value])

        # Calculate angle between right shoulder, elbow and wrist to looking for straight right arm
        right_arm_angle = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value])

        #  Calculate angle between right arm and hip
        right_arm_hip_angle = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value])

        # Calculate angle between right hip, knee, and ankle
        right_leg_angle = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value])

        # find if hands are between hips
        left_hand_inside = is_between(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
                                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])

        right_hand_inside = is_between(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
                                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
        inside = left_hand_inside and right_hand_inside

        right_leg_in = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE][0] > landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP][0]


        if 200 < left_arm_angle < 220 and 130 < right_arm_angle < 150 and inside:
            playsound('1.wav')

        if 210 < right_arm_angle < 250 and 240 < right_arm_hip_angle < 270 and not right_hand_inside:
            playsound('2.wav')

        if 80 < right_leg_angle < 110 and right_leg_in:
            playsound('3.wav')



def main():
    global detector
    global landmarks
    global p_time
    cap = cv2.VideoCapture(0)
    while True:
        # read image from video feed
        success, img = cap.read()
        img = cv2.flip(detector.pose_init(img), 1)

        landmarks = detector.find_landmarks(img, draw=False)
        # print(lm_list)

        find_mj()
        # calculate frame rate
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # display video
        cv2.putText(img, str(int(fps)) + ' fps', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('King of Pop', img)

        # control speed
        cv2.waitKey(1)


main()
