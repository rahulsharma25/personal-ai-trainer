import cv2 as cv
import numpy as np 
import mediapipe as mp 
import time
import pose_estimation_module as pem

def main():
    cap = cv.VideoCapture(0)
    pose_detector = pem.Pose_estimator(min_detection_confidence=0.8)
    p_time = time.time()

    count_right_hand_reps = 0   
    count_left_hand_reps = 0
    
    MAX_ANKLE_ANGLE = 155
    MIN_ANKLE_ANGLE = 40

    right_hit_bottom, right_hit_top = False, False
    left_hit_bottom, left_hit_top = False, False

    while True:
        isTrue, img = cap.read()

        img = pose_detector.find_pose(img, draw=False)
        lm_list = pose_detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            ####################################################################################################################
            right_hand_angle = pose_detector.find_angle(img, 12, 14, 16)
            right_hand_meter_height = np.interp(right_hand_angle, [MIN_ANKLE_ANGLE, MAX_ANKLE_ANGLE], [50, 200])
            cv.rectangle(img, (30, 200), (60, int(right_hand_meter_height)), (0, 0, 255), -1)

            if right_hit_bottom and right_hit_top:
                count_right_hand_reps += 0.5
                right_hit_bottom = False
                right_hit_top = False

            if right_hand_angle >= MAX_ANKLE_ANGLE:
                right_hit_bottom = True
            if right_hand_angle <= MIN_ANKLE_ANGLE:
                right_hit_top = True

            ####################################################################################################################
            left_hand_angle = pose_detector.find_angle(img, 11, 13, 15)
            left_hand_meter_height = np.interp(left_hand_angle, [MIN_ANKLE_ANGLE, MAX_ANKLE_ANGLE], [50, 200])
            cv.rectangle(img, (580, 200), (610, int(left_hand_meter_height)), (0, 0, 255), -1)

            if left_hit_bottom and left_hit_top:
                count_left_hand_reps += 0.5
                left_hit_bottom = False
                left_hit_top = False

            if left_hand_angle >= MAX_ANKLE_ANGLE:
                left_hit_bottom = True
            if left_hand_angle <= MIN_ANKLE_ANGLE:
                left_hit_top = True

            ####################################################################################################################

        img = cv.flip(img, 1)
    
        cv.rectangle(img, (580, 200), (610, 50), (0, 0, 255), 2)
        cv.putText(img, str(int(count_right_hand_reps)), (585, 230), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        cv.rectangle(img, (30, 200), (60, 50), (0, 0, 255), 2)
        cv.putText(img, str(int(count_left_hand_reps)), (35, 230), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        cv.putText(img, f'FPS: {int(fps)}', (10, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv.imshow("Video", img)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    return

if __name__ == "__main__":
    main()