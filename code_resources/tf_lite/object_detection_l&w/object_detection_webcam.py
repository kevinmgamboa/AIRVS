"""
Webcam Object Detection
Created on Sun Feb  9 09:26:38 2020
@author: Kevin Machado Gamboa
"""
import cv2

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from visualization_utils import draw_bounding_boxes_on_image_array
from utils import reshape_image
from detector import ObjectDetectorLite


# -----------------------------------------------------------------------------
#                                  Main Code
# -----------------------------------------------------------------------------
def main(model_path='detect.tflite', label_path='labelmap.txt', confidence=0.5):
    """
    Main function to initialize the object detector and start video capture.

    Parameters:
    model_path (str): Path to the Tflite model.
    label_path (str): Path to the model labels.
    confidence (float): Minimum required confidence level of bounding boxes.
    """
    # Load & initialize model
    detector = ObjectDetectorLite(model_path=model_path, label_path=label_path)
    input_size = detector.get_input_size()
    print(f"model input image size: {input_size}")

    plt.ion()
    plt.tight_layout()

    fig = plt.gcf()
    fig.suptitle('Detecting')
    ax = plt.gca()
    ax.set_axis_off()

    cap = cv2.VideoCapture(0)
    while True:
        # reads image from camera
        ret, image_np = cap.read()
        # reshapes image to lower dimension
        image = reshape_image(image_np)
        # passes image into model
        boxes, scores, classes = detector.detect(image, confidence)
        # draw box into image
        if len(boxes) > 0:
            draw_bounding_boxes_on_image_array(image, boxes, display_str_list=classes)

        # show the output frame
        cv2.imshow('object detection', cv2.resize(image_np, (680, 640)))

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
