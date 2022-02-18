import cv2 as cv
import mediapipe as mp
import time
import math

class Pose_estimator():
    def __init__(self, mode=False, complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity,
                                        self.smooth_landmarks,
                                        self.enable_segmentation,
                                        self.smooth_segmentation,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
                
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_pose(self, img, draw=True):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.pose.process(rgb_img)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for i, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lm_list.append([i, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 7, (255, 0, 255), 2)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        sx1, sy1 = x1-x2, y1-y2
        sx3, sy3 = x3-x2, y3-y2
        sx2, sy2 = 0, 0

        angle = abs(math.atan2(sy1, sx1) - math.atan2(sy3, sx3))
        angle = math.degrees(angle)
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.line(img, (x2, y2), (x3, y3), (0, 255, 255), 2)
            cv.circle(img, (x1, y1), 5, (0, 0, 255), -1)
            cv.circle(img, (x1, y1), 8, (0, 0, 255), 1)
            cv.circle(img, (x2, y2), 5, (0, 0, 255), -1)
            cv.circle(img, (x2, y2), 8, (0, 0, 255), 1)
            cv.circle(img, (x3, y3), 5, (0, 0, 255), -1)
            cv.circle(img, (x3, y3), 8, (0, 0, 255), 1)

        return int(angle)

def main():
    pose_detector = Pose_estimator()
    cap = cv.VideoCapture(0)

    p_time = 0

    while True:
        isTrue, frame = cap.read()

        img = pose_detector.find_pose(frame)
        lm_list = pose_detector.find_position(frame)

        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        cv.putText(img, str(int(fps)), (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

        cv.imshow("Video", frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()