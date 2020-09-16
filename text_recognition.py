# ToDo: make it global
"""
this project intends to color the text of transistors in an analog circuit.
to perform such a method I've used open-cv with OCR.
first I've used open-cv deep learning paper (Natural scene text understanding) by Céline Mancas-Thillou, Bernard Gosselin.
which showed great success in text localization.
So I'm using it, and I've followed this tutorial with some edits, to fit my project.
to make it happens (https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) (I'm c++ fan ^_^).
after localizing the text, I've performed OCR to detect the words.
If i've used OCR on its own, it should great failure in noisy image, thats why I've preferred to use EAST with it.
*** I've made some changes to make a better localization, detection. it still have very small errors.
features:
    1- you can select whatever the transistor you like from a drop down list.
    2- transistor selected will be colored.
    3- simple GUI.

limitations:
    1- if there were many words, or complex structure, it will fail detecting the right place, and words.
"""

from PySide2.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QComboBox
import sys
from PySide2.QtGui import QPixmap
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2


def decode_predictions(scores, geometry):
    # get the number of rows and columns from the scores.
    (numRows, numCols) = scores.shape[2:4]
    # initialize set of bounding box rectangles and corresponding
    # confidence scores
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # if scoresData[x] < args["min_confidence"]:
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


# input image
image = cv2.imread("images/Try.PNG")
# make a copy and get its shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (32 * 20, 32 * 20)
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# We used non-maxima suppression as the NMS provided by Python open-cv doesn't work (with me at least).
# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * 0.1)
    dY = int((endY - startY) * 0.1)

    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)

    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r: r[0][1])

output = orig.copy()


# loop over the results
def drawRectAndColoring(selecteditem):
    for ((startX, startY, endX, endY), text) in results:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        if text == selecteditem:
            if selecteditem == "Faiviia wv":
                text = "M1a"
            elif selecteditem == "la mibpo V,":
                text = "M2a"
            cv2.rectangle(output, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(output, text, (startX + 2, endY - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)


# main layout
class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Schematic Plotter"
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        vbox = QVBoxLayout()

        # make a label to let the user know what is he/she using.
        self.labelImage = QLabel(self)
        self.label_text = QLabel("Select Your Transistor")

        # add image to the label, so the user can see it.
        self.pixmap = QPixmap("Try.PNG")
        self.labelImage.setPixmap(self.pixmap)

        # make a drop down list, add names to it.
        self.box = QComboBox()
        self.box.addItem("M1")
        self.box.addItem("M2")
        self.box.addItem("M3")
        self.box.addItem("M4")
        self.box.addItem("M5")
        self.box.addItem("M6")

        # on selection, call the function.
        self.box.currentTextChanged.connect(self.getText)

        # add widgets, and set the layout.
        vbox.addWidget(self.label_text)
        vbox.addWidget(self.box)
        vbox.addWidget(self.labelImage)
        self.setLayout(vbox)
        self.show()

    def getText(self, i):
        # I've needed to pass certain names whenever the user selects an item,
        # so I can highlight the selected item only.
        global output
        if i == "M1":
            drawRectAndColoring("Faiviia wv")
            drawRectAndColoring("la mibpo V,")
        elif i == "M2":
            drawRectAndColoring("M2a_")
            drawRectAndColoring("M2b")
        elif i == "M3":
            drawRectAndColoring("M3a _")
            drawRectAndColoring("M3b")
        elif i == "M4":
            drawRectAndColoring("M4a")
            drawRectAndColoring("M4b")
        elif i == "M5":
            drawRectAndColoring("M5a")
            drawRectAndColoring("M5b")
        elif i == "M6":
            drawRectAndColoring("M6")

        # After colorization, I'll save the output produced, so I can embed it in another layout.
        cv2.imwrite("Text Detection.PNG", output)
        self.w = SecondWindow()
        self.w.show()
        output = orig.copy()


class SecondWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Schematic Plotter"
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        vbox = QVBoxLayout()
        labelImage = QLabel(self)
        pixmap = QPixmap("Text Detection.PNG")
        labelImage.setPixmap(pixmap)
        vbox.addWidget(labelImage)
        self.setLayout(vbox)
        self.show()


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    Plotter = Window()
    #    Plotter.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
