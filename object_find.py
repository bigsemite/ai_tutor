import cv2
#The datasheet of this model can be found here: https://docs.openvinotoolkit.org/2019_R1/_person_vehicle_bike_detection_crossroad_0078_description_person_vehicle_bike_detection_crossroad_0078.html

model = cv2.dnn.readNetFromCaffe('models/person/deploy_per.prototxt', 'models/person/person-vehicle-bike-detection-crossroad-0078.caffemodel')
classes = ['nothing','person','Vehicle', 'bike']

def getImg():
    vd = input("Enter the image url: ")
    frame = cv2.imread(vd)
    frame = findPerson(frame)
    while True:
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) == 27:
            break
    return

def getIn():
    
    c = input('Enter the Camera Url:')
    if c =='0':
        c = 0
    cap = cv2.VideoCapture(c)
    if cap.isOpened() == False:
        print("Error in opening video stream or file")
    while(cap.isOpened()):
        status, frame = cap.read()
        findPerson(frame)
            # Display the resulting frame
        cv2.imshow('Frame',frame)
            # Press esc to exit
        if cv2.waitKey(2) & 0xFF == 27:
            break
    return

def findPerson(img):
    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,1.0,(300,300),(104, 117, 123))
    
    model.setInput(blob)
    result = model.forward()
    useful_result = result[0][0]
    for i in range(len(useful_result)):
        cl = int(useful_result[i][1])
        confd = useful_result[i][2]
        if cl > 0:
            if confd >= 0.5:
                print (classes[cl],'found with confidence ', confd)
                startX = int(useful_result[i][3] * w)
                startY = int(useful_result[i][4] *h)
                endX  = int(useful_result[i][5] * w)
                endY = int(useful_result[i][6] * h)
                print ('AT: ', startX, startY,endX, endY)               
                #draw on the image
                cv2.rectangle(img,(startX,startY),(endX, endY),(0,255,0),2)                   #Color is by default black
                f = '{} {:.2f}'.format(classes[cl], confd * 100) 
                
                cv2.putText(img, f, (startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
   # print (useful_result)
    #exit()
    return img


if __name__ == "__main__":
    getIn()
    