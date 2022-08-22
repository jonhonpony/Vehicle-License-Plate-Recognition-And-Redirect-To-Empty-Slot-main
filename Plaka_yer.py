import cv2
import numpy as np
#import plaka_okuma

def detection(img):
    img2=img
    w=img.shape[1]
    h=img.shape[0]
    best_confidence=[]
    best_start_x=[]
    best_start_y=[]
    best_end_x=[]
    best_end_y=[]
    img_blob=cv2.dnn.blobFromImage(img, 1/255, (416,416),swapRB=True,crop=False)
    labels=["plate"]
    colors=[0,255,255]
    #colors=[np.array(color.split(",")).astype("int") for color in colors]

    model=cv2.dnn.readNetFromDarknet("plate_yolov4.cfg",darknetModel="plate_yolov4_best.weights")
    layers=model.getLayerNames()
    output_layer=[ layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(img_blob)
    detection_layers=model.forward(output_layer)

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores=object_detection[5:]
            predicted_id=np.argmax(scores)
            confidence =scores[predicted_id]


            if confidence > 0.30:
                label=layers[predicted_id]
                bounding_box=object_detection[0:4]*np.array([w,h,w,h])
                (box_center_x,box_center_y,box_w,box_h)=bounding_box.astype("int")
                start_x=int(box_center_x-(box_w/2))
                start_y=int(box_center_y-(box_h/2))
                best_start_x.append(start_x)
                best_start_y.append(start_y)
                end_x=start_x+box_w
                end_y=start_y+box_h
                best_end_x.append(end_x)
                best_end_y.append(end_y)
                best_confidence.append(float(confidence))
                box_color=colors#colors[predicted_id]
                #box_color=[int(each) for each in box_color]
                ###cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
                #cv2.putText(img, label, (start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], int(1))

    max=np.argmax(best_confidence)#en büyük oranlıyı buldum hangi sırada onu bulucam
    #print(max)
    cv2.rectangle(img2,(best_start_x[max],best_start_y[max]),(best_end_x[max],best_end_y[max]),[255,0,0],1)
    cropped=img2[best_start_y[max]:best_end_y[max]+1,best_start_x[max]+1:best_end_x[max]]
    #cv2.imshow("dassa",cropped)
    """
    gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    mask=np.zeros((gray.shape),np.uint8)
    (x,y)=np.where(img2==[255,0,0])#Beyaz olan yerlerin konumlarını alıyoruz
    (topx,topy)=(np.min(x),np.min(y))#minimum değerleri alıyoruz
    (bottomx,bottomy)=(np.max(x),np.max(y))#maksimum değerleri alıyoruz
    cropped=img2[topx+1:bottomx,topy:bottomy+1]#plakanın olduğu yeri kesiyoruz
    print(best_start_x[max])
    cv2.imshow("dassa",cropped)
    """
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow("img2",img2)
    cv2.waitKey(100)
    return cropped
"""
img=cv2.imread("c.png")
img=detection(img)
#cv2.imwrite("plaka.jpg",img)
#a=plaka_okuma.ayirma(img,img)
#print(a)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret,img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
plaka_okuma.hazir(img)
"""

