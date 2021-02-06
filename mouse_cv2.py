import cv2

img = None
pr = 0
px=0
py=0

def getIn():
    global img
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('result1')
    cv2.setMouseCallback('result1', getMouse)
    tp = px
    while True:
        _,img = cap.read()
        cv2.putText(img, 'State: ' + str(pr), (20,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        if px != tp :
            cv2.circle(img,(px,py),50,(0,0,255),-1)
            tp = px
        cv2.imshow('result1', img)
        if cv2.waitKey(1) == 27: break
    return

def getMouse(event, x,y,flags, params):
    global pr, px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Clicked')
        pr += 1
        px=x
        py = y
        
    
if __name__ == "__main__":
    getIn()