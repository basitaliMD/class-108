import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#detection will detect hand and tracking will track hand
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

tip_IDs = [4, 8, 12, 16, 20]

def draw_hand_landmarks(frame, hand_landmarks):
    #draw collection between landmark points
    if(hand_landmarks == True):
        for i in hand_landmarks:
            mp_drawing.draw_landmarks(frame, i , mp_hands.HAND_CONNECTIONS)
            
def count_fingers(frame, hand_landmarks, handNo = 0):
    if(hand_landmarks == True):
        landmarks = hand_landmarks[handNo].landmark
        fingers = []
        for index in tip_IDs:
            finger_tip_y = landmarks[index].y
            finger_bottom_y = landmarks[index-2].y
            if(index != 4):
                if(finger_tip_y < finger_bottom_y):
                    fingers.append(1)
                    print(index, "isOpen")
                if(finger_tip_y > finger_bottom_y):
                    fingers.append(0)
                    print(index, "isClosed")
        total_fingers = fingers.count(1)
        text = f"fingers:{total_fingers}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

while True:
    flag, frame = capture.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(frame)
    #get landmark position from result
    hand_landmarks = results.multi_hand_landmarks
    #call the function and draw the landmark points
    draw_hand_landmarks(frame, hand_landmarks)
    #get fingers Position
    count_fingers(frame, hand_landmarks)
    cv2.imshow("media", frame)
    key = cv2.waitKey(1)
    if(key == 32):
        break
    
cv2.destroyAllWindows()