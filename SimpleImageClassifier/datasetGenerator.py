import cv2

capture=cv2.VideoCapture(0)
n=1
activation=False
while n<=100:
    _, frame = capture.read()
    cv2.imshow('video',frame)

    if cv2.waitKey(3) == ord('s'):
        print('activated')
        activation=True
    if activation == True:
        cv2.imwrite('SomethingOrNothing/Nothing/nothing'+str(n)+'.png',frame)
        n+=1

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
