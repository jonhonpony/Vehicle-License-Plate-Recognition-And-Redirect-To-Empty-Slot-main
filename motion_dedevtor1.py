import cv2
import numpy as np
import pyautogui
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
bos=[]
dolu=[]
class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1


    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.durum = []

    def detect_motion(self):
        cv2.imread(self.video)

        coordinates_data = self.coordinates_data

        for p in coordinates_data:
            coordinates = self._coordinates(p)

            rect = cv2.boundingRect(coordinates)#aga bu x ve y değerlerinin en küçük olanını ve enbüyük olanlarla farkını alır

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = cv2.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=cv2.LINE_8)

            mask = mask == 255
            self.mask.append(mask)

        statuses = [False] * len(coordinates_data)

        times = [None] * len(coordinates_data)
        i=0
        durum = []
        j=0
        frame=cv2.imread(self.video)


        blurred = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
        grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


        if j==0:
            j=1
            for index, c in enumerate(coordinates_data):
                dolu.append((index+1))
                bos.append((index+1))
            print(dolu)
            print(bos)
        for index, c in enumerate(coordinates_data):

            status = self.__apply(grayed, index, c)


            if i==0:
                durum.append(status)
                if status==False:
                    print((index+1),". park yeri dolu")
                    a=index+1
                    bos.remove(a)

                if status==True:
                    print((index+1),". park yeri boş")
                    a=index+1
                    dolu.remove(a)

            if durum[index] !=status:
                if status==False:
                    print((index+1),". park yeri dolu")
                    a=index+1
                    dolu.append(a)
                    bos.remove(a)
                if status==True:
                    print((index+1),". park yeri boş")
                    a=index+1

                    bos.append(a)

                    dolu.remove(a)

                durum[index]=status

        if i==0:
            i=1
        cv2.destroyAllWindows()
        min=np.argmin(bos)
        print(str(bos[min])+" nolu alana geçebilirsin")
        pyautogui.alert(text=str(bos[min])+" nolu alana geçebilirsin", title='Bos Alan', button='OK')

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)

        rect = self.bounds[index]

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])


class CaptureReadError(Exception):
    pass
