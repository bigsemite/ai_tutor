import cv2


md = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
tx = 'models/face_detector/deploy.prototxt'
model = cv2.dnn.readNetFromCaffe(tx,md)

def getImg():
    vd = input("Enter the image url: ")
    frame = cv2.imread(vd)
    frame = findObj(frame)
    while True:
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) == 27:
            break
    return
    
def getIn():
    print('Enter the Camera URL: ')
    c = input()
    if c == '0': c=0
    cap = cv2.VideoCapture(c)
    while True:
        status, frame = cap.read()
        frame = findObj(frame)
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) == 27: break
    return

def findObj(img):
    (h,w) = img.shape[:2]
    #img1 = cv2.resize(img,(300,300))
    blob = cv2.dnn.blobFromImage(img,1.0,(300,300),(104, 117, 123)) 
    
        #Creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
        ##Pramaters
        #image	input image (with 1-, 3- or 4-channels).
        #size	spatial size for output image
        #mean	scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
        #scalefactor	multiplier for image values.
        #swapRB	flag which indicates that swap first and last channels in 3-channel image is necessary.
        #crop	flag which indicates whether image will be cropped after resize or not
        #ddepth	Depth of output blob. Choose CV_32F or CV_8U.
    model.setInput(blob)
    result = model.forward()
    useful_result = result[0][0]
    
    print(len(useful_result))
    #loop through the results an find high confidence
    for i in range(0, len(useful_result)):
        confidence = useful_result[i][2]
        if (confidence > 0.5 ):
            #if found, get the coordinates of the faces
            startX = int(useful_result[i][3] * w)
            startY = int(useful_result[i][4] * h)
            endX = int(useful_result[i][5] * w)
            endY = int(useful_result[i][6]*h)
            print('Face found at ', startX,startY,endX,endY)
            #draw rectange to around the faces
            face = img[startY:endY, startX:endX]
            #if face:
            cv2.imwrite('face.jpg', face)
            cv2.rectangle(img,(startX, startY),(endX, endY),(0,0,255),1)
            #write something on the face
            cf = "{:.2f}%".format(confidence * 100)
            cv2.putText(img,cf, (endX, endY+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)                  
    return img

if __name__ == "__main__":
    getIn()
    #getImg()
    
    
    