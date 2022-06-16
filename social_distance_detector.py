# import the necessary packages
import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import os
import time


from pygame import mixer
mixer.init()
sound = mixer.Sound('one.wav')
alert = mixer.Sound('two.wav')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.path, "yolov3.weights"])
configPath = os.path.sep.join([config.path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("STARTING STREAM...")
print("LOADING YOLO FROM DISK..")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("ACCESSING VIDEO STREAM...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 1)
fps = FPS().start()
writer = None

sound.play()

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # resize the frame and then detect people (and only people) in it 700
    frame = imutils.resize(frame, width=800)

    results = detect_people(frame, net, ln,
                            personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social
    # distance
    violate = set()
    #colors = findcolor(frame)

    # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.min_distance:
                    # update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        # if the index pair exists within the violation set, then
        # update the color
        if i in violate:
            color = (0, 0, 255)
            alert.play()
            time.sleep(2)

        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # BLUE
        lower_blue = np.array([40, 3, 255])
        upper_blue = np.array([110, 150, 255])
        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        for c in cnts2:
            area2 = cv2.contourArea(c)
            if area2 > 2000:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "BLUE", (cx-20, cy-20),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 3)
                break
        # RED
        lower_red = np.array([0, 50, 120])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 2000:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "RED", (cx-20, cy-20),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 3)
                break
        # YELLOW
        lower_yellow = np.array([25, 70, 120])
        upper_yellow = np.array([30, 255, 255])
        mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts3 = imutils.grab_contours(cnts3)
        for c in cnts3:
            area3 = cv2.contourArea(c)
            if area3 > 2000:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "YELLOW", (cx-20, cy-20),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 3)
                break
        # GREEN
        lower_green = np.array([100, 255, 80])
        upper_green = np.array([70, 255, 255])
        mask4 = cv2.inRange(hsv, lower_green, upper_green)
        cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts4 = imutils.grab_contours(cnts4)
        for c in cnts4:
            area4 = cv2.contourArea(c)
            if area4 > 2000:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "GREEN", (cx-20, cy-20),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 3)
                break
    # draw the total number of social distancing violations on the
    # output frame
    text = "Violators Count : {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    # (0,0,255)

    # check to see if the output frame should be displayed to our
    # screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
