# pip install face_recognition
#
import face_recognition
import cv2

load1 = face_recognition.load_image_file('semiu.jpg')
load2= face_recognition.load_image_file('kosy1.jpg')
load3 = face_recognition.load_image_file('kosy2.jpg')
pic = [face_recognition.face_encodings(load1)[0],
       face_recognition.face_encodings(load2)[0],
       face_recognition.face_encodings(load3)[0]]
name = ['Semiu','Kosy','Kosy']

def getIn():
    p = input('Enter the Camera URL:')
    if p == '0':
        p = 0
    cap = cv2.VideoCapture(p)
    while True:
        status, frame = cap.read()
        frame = recog(frame)
        cv2.imshow('Result', frame)
        if cv2.waitKey(2) == 27: break

def recog(img):
    #convert the image to rgb
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #find faces in the given image
    loc = face_recognition.face_locations(img1)
    #extract the found faces
    found = face_recognition.face_encodings(img1, loc)
    
    #compare the found
    for (top,right, bottom, left), fc in zip(loc, found):
        seen_face ='Unknown'
        
        #compare each face with known faces
        matches = face_recognition.compare_faces(pic, fc, tolerance=0.5)
        
        if True in matches:
            m_index = matches.index(True)
            seen_face = name[m_index]
        
        print (seen_face)
        #show rectangle and name
        cv2.rectangle(img, (left,top),(right, bottom), (0,255,0),2)
        cv2.putText(img, seen_face, (left, bottom + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    return img
       
if __name__ == "__main__":
    getIn()