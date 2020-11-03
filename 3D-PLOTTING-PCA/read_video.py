import cv2

cap = cv2.VideoCapture('vid.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
left = cv2.VideoWriter('fish.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width // 2, frame_height))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        left.write(frame[:, frame_width // 2:frame_width])
        cv2.imshow('frame', frame[:, frame_width // 2:frame_width])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
left.release()
cv2.destroyAllWindows()
