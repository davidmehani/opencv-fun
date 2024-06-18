import cv2
import mediapipe as mp
import time

# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker pose reference

class PoseDetector:
    def __init__(self, mode=False, upper_body=False, smooth=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upper_body, self.smooth,
                                      min_detection_confidence=self.detection_confidence,
                                      min_tracking_confidence=self.tracking_confidence)

    def pose_init(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process feed into pose recognition
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_landmarks(self, img, draw=True):
        landmarks = {}

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # get absolute coordinates
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks[id] = [cx, cy]
                # overlay circle on coordinates to show it working
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = PoseDetector()
    while True:
        # read image from video feed
        success, img = cap.read()
        img = cv2.flip(detector.pose_init(img), 1)

        landmarks = detector.find_landmarks(img, draw=False)
        print(landmarks)

        # calculate frame rate
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # display video
        cv2.putText(img, str(int(fps)) + ' fps', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)

        # control speed
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
