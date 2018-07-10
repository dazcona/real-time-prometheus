# https://github.com/santiagxf/prometheus/blob/master/FireDetection/utils/plot_helpers.py

from __future__ import print_function
import sys
import numpy as np
import copy, textwrap
from PIL import Image, ImageFont, ImageDraw
import cv2


def visualize_detections(img_path, roi_coords, roi_labels, roi_scores,
                         pad_width, pad_height, classes,
                         draw_negative_rois=False, decision_threshold=0.0):

    # read and resize image
    imgWidth, imgHeight = imWidthHeight(img_path)
    scale = 800.0 / max(imgWidth, imgHeight)
    imgHeight = int(imgHeight * scale)
    imgWidth = int(imgWidth * scale)
    if imgWidth > imgHeight:
        h_border = 0
        v_border = int((imgWidth - imgHeight) / 2)
    else:
        h_border = int((imgHeight - imgWidth) / 2)
        v_border = 0

    PAD_COLOR = [255, 255, 255]
    cv_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (imgWidth, imgHeight), interpolation=cv2.INTER_NEAREST)
    result_img = cv2.copyMakeBorder(resized, v_border, v_border, h_border, h_border, cv2.BORDER_CONSTANT,
                                    value=PAD_COLOR)
    rect_scale = 800 / pad_width

    assert (len(roi_labels) == len(roi_coords))
    if roi_scores is not None:
        assert (len(roi_labels) == len(roi_scores))
        minScore = min(roi_scores)
        if minScore > decision_threshold:
            decision_threshold = minScore * 0.5

    # draw multiple times to avoid occlusions
    for iter in range(0, 3):
        for roiIndex in range(len(roi_coords)):
            label = roi_labels[roiIndex]
            if roi_scores is not None:
                score = roi_scores[roiIndex]
                if decision_threshold and score < decision_threshold:
                    label = 0

            # init drawing parameters
            thickness = 1
            if label == 0:
                color = (255, 0, 0)
            else:
                color = getColorsPalette()[label]

            rect = [(rect_scale * i) for i in roi_coords[roiIndex]]
            rect[0] = int(max(0, min(pad_width, rect[0])))
            rect[1] = int(max(0, min(pad_height, rect[1])))
            rect[2] = int(max(0, min(pad_width, rect[2])))
            rect[3] = int(max(0, min(pad_height, rect[3])))

            # draw in higher iterations only the detections
            if iter == 0 and draw_negative_rois:
                drawRectangles(result_img, [rect], color=color, thickness=thickness)
            elif iter == 1 and label > 0:
                thickness = 4
                drawRectangles(result_img, [rect], color=color, thickness=thickness)
            elif iter == 2 and label > 0:
                font = ImageFont.load_default()
                text = str(classes[label])
                if roi_scores is not None:
                    text += "(" + str(round(score, 2)) + ")"
                result_img = drawText(result_img, (rect[0], rect[1]), text, color=(255, 255, 255), font=font,
                                      colorBackground=color)

    return result_img


def imWidthHeight(input):
    width, height = Image.open(input).size #this does not load the full image
    return width, height


def getColorsPalette():
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
    for i in range(5):
        for dim in range(0,3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    newColor = copy.deepcopy(colors[i])
                    newColor[dim] = int(round(newColor[dim] * s))
                    colors.append(newColor)
    return colors

def drawRectangles(img, rects, color = (0, 255, 0), thickness = 2):
    for rect in rects:
        pt1 = tuple(ToIntegers(rect[0:2]))
        pt2 = tuple(ToIntegers(rect[2:]))
        try:
            cv2.rectangle(img, pt1, pt2, color, thickness)
        except:
            print("Unexpected error:", sys.exc_info()[0])


def ToIntegers(list1D):
    return [int(float(x)) for x in list1D]


def drawText(img, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = None):
    pilImg = imconvertCv2Pil(img)
    pilImg = pilDrawText(pilImg,  pt, text, textWidth, color, colorBackground, font)
    return imconvertPil2Cv(pilImg)


def imconvertCv2Pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)


def pilDrawText(pilImg, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = None):
    textY = pt[1]
    draw = ImageDraw.Draw(pilImg)
    if textWidth == None:
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=textWidth)
    for line in lines:
        width, height = font.getsize(line)
        if colorBackground != None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(colorBackground[::-1]))
        draw.text(pt, line, fill = tuple(color), font = font)
        textY += height
    return pilImg


def imconvertPil2Cv(pilImg):
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]

