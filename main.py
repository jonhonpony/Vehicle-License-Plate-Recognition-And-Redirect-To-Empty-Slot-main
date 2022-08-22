import cv2
import Plaka_yer
import plaka_okuma
import pyautogui
import plate
kameragiris = cv2.VideoCapture(2)
i=0
while i==0:
    result =pyautogui.prompt(text='Arac Geldiyse y e basınız cikmak için e basiniz', title='Araç Kontrolü')
    if result=="y":
        ret, img = kameragiris.read()
        img=Plaka_yer.detection(img)
        a=plaka_okuma.ayirma(img,img)
        kameragiris.release()
        plate.slot()
    if result=="e":
        i=1

kameragiris.release()
cv2.destroyAllWindows()