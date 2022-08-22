import cv2
import numpy as np
from drawing_utils import draw_contours

COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)

class CoordinatesGenerator:
    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")

    def __init__(self, image, output):

        self.output = output
        self.caption = image
        self.color = COLOR_BLUE

        self.image = cv2.imread(image).copy()
        self.click_count = 0
        self.ids = 0
        self.coordinates = []

        cv2.namedWindow(self.caption, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.caption, self.__mouse_callback)

    def generate(self):#tuşa basıldı mı kontrol ediyor
        while True:

            cv2.imshow(self.caption, self.image)
            key = cv2.waitKey(0)

            if key == CoordinatesGenerator.KEY_RESET:
                self.image = self.image.copy()
            elif key == CoordinatesGenerator.KEY_QUIT:
                break
        cv2.destroyWindow(self.caption)

    def __mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:#herhangi bi yere basıldığında gir
            self.coordinates.append((x, y))#basıldığı yerleri kaydet
            self.click_count += 1#kaç defa basıldığına bakar

            if self.click_count >= 4:
                self.__handle_done()

            elif self.click_count > 1:
                self.__handle_click_progress()

        cv2.imshow(self.caption, self.image)

    def __handle_click_progress(self):#1 den fazla çizgi varsa arasını çiziyor

        cv2.line(self.image, self.coordinates[-2], self.coordinates[-1], (255, 0, 0), 1)

    def __handle_done(self):#4 yer alındıysa kare çiz

        cv2.line(self.image,
                     self.coordinates[2],
                     self.coordinates[3],
                     self.color,
                     1)
        cv2.line(self.image,
                     self.coordinates[3],
                     self.coordinates[0],
                     self.color,
                     1)

        self.click_count = 0

        coordinates = np.array(self.coordinates)

        self.output.write("-\n          id: " + str(self.ids) + "\n          coordinates: [" +
                          "[" + str(self.coordinates[0][0]) + "," + str(self.coordinates[0][1]) + "]," +
                          "[" + str(self.coordinates[1][0]) + "," + str(self.coordinates[1][1]) + "]," +
                          "[" + str(self.coordinates[2][0]) + "," + str(self.coordinates[2][1]) + "]," +
                          "[" + str(self.coordinates[3][0]) + "," + str(self.coordinates[3][1]) + "]]\n")

        draw_contours(self.image, coordinates, str(self.ids + 1), COLOR_WHITE)

        for i in range(0, 4):
            self.coordinates.pop()

        self.ids += 1
