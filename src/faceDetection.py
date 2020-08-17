import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        #cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.2, 10, color['blue'], "Face")    
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img, eyeCascade, 1.2, 14, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.4, 10, color['green'], "Nose")
        coords = draw_boundary(roi_img, mouthCascade, 1.7, 15, color['white'], "Mouth")
    return img

faceCascade = cv2.CascadeClassifier(r'C:\Users\Rahul\Pictures\faceRecg\haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier(r'C:\Users\Rahul\Pictures\faceRecg\haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier(r'C:\Users\Rahul\Pictures\faceRecg\Nariz.xml')
mouthCascade = cv2.CascadeClassifier(r'C:\Users\Rahul\Pictures\faceRecg\Mouth.xml')

video_capture = cv2.VideoCapture(0)
while True:
    check, img = video_capture.read()        
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

video_capture.release()
cv2.destroyAllWindows()
