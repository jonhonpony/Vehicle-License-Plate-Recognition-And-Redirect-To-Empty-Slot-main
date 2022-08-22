import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import numpy as np
import pyautogui
import pytesseract
MIN_PIXEL_WIDTH = 5
MIN_PIXEL_HEIGHT = 8
b=0
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

sifir=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
bir=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
iki=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
üc=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
dort=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
bes=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
alti=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
yedi=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
sekiz=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
dokuz=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
a=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
b=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
c=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
d=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
e=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
f=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
g=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
h=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
i=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
j=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
k=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
l=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
m=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
n=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
o=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
p=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
r=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
s=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
t=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
u=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
v=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
y=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
z=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

model = keras.models.load_model("my_model.h5")
def plate_inv(img):

    listOfPossiblePlates = []                   # this will be the return value

    #imgContours = np.zeros((height, width, 3), np.uint8)

    height, width, numChannels = img.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    imgGrayscale = imgValue
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)

    imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5, 5), 0)

    imgThreshScene = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    cv2.imshow("dsa", imgThreshScene)
    return imgThreshScene
def bulma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    data = np.array([])
    cikis=''
    data=np.append(data,img)
    data=np.reshape(data, (-1,24,24,1))
    accuracy=0.9
    giris=np.reshape(sifir,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='0'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(bir,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='1'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(iki,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='2'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(üc,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='3'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(dort,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='4'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(bes,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='5'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(alti,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='6'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(yedi,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='7'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(yedi,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='7'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(sekiz,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='8'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(dokuz,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='9'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(a,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='A'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(b,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='B'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(c,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy :
        accuracy=result[1]
        cikis='C'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(d,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='D'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(e,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='E'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(f,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='F'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(g,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='G'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(h,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='H'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(i,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='I'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(j,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='J'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(k,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='K'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(l,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='L'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(m,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='M'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(n,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='N'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(o,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='o'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(p,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='P'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(r,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='R'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(s,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='S'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(t,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='T'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(u,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='U'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(v,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='V'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(y,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='Y'
    if accuracy>0.9:
        return cikis, accuracy
    giris=np.reshape(z,(1,32))
    result=model.evaluate(data,giris,verbose=2)
    if result[1]>accuracy:
        accuracy=result[1]
        cikis='Z'
    if accuracy>0.9:
        return cikis, accuracy
    return cikis, accuracy

def ayirma(img,imgorjinal):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #img=plate_inv(img=img)
    contours, npaHierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#konturu alıyoruz

    height = img.shape[0]
    width = img.shape[1]
    kernel = np.ones((3,3),np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    listofpossbile=[]
    strChars = ""
    for contour in contours:#bulunan konturların arasında harflerin konumları bulunur.

        boundingRect = cv2.boundingRect(contour)
        [intX, intY, intWidth, intHeight] = boundingRect
        intBoundingRectArea=intWidth*intHeight
        AspectRatio=intWidth/intHeight

        if (intBoundingRectArea > MIN_PIXEL_AREA and
            intWidth > MIN_PIXEL_WIDTH and intHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < AspectRatio and AspectRatio < MAX_ASPECT_RATIO):
            #print(boundingRect)
            listofpossbile.append(boundingRect)

    a=0
    for i in range(0,len(listofpossbile)):#bulunan harfler y eksenine göre sıralama yapılır
        for j in range(0,len(listofpossbile)-1):
            if listofpossbile[j][0] > listofpossbile[j+1][0] :
                a=listofpossbile[j]
                listofpossbile[j]=listofpossbile[j+1]

                listofpossbile[j+1]=a
            else:
                continue
    b=0
    for boundingRect in listofpossbile:#harfleri tek tek okumaya gönderir
        [intX, intY, intWidth, intHeight] = boundingRect
        imgROI = imgorjinal[(intY-2) : (intY+intHeight+4),(intX) : (intX + intWidth)]
        imgROIResized = cv2.resize(imgROI, (24, 24))
        strCurrentChar='0'
        (cikis,a)=bulma(imgROIResized)
        if b==0:
            b=1
            strCurrentChar = str(cikis)
            strChars = strChars + strCurrentChar
        if cikis=='0' and strCurrentChar=='0':
            continue
        strCurrentChar = str(cikis)
        strChars = strChars + strCurrentChar

    pyautogui.alert(text="Aracın Plakası "+strChars, title='Plaka', button='OK')
    return strChars



