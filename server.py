# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
#from imutils import build_montages
from datetime import datetime
from imutils.video import FPS
import numpy as np
from imagezmq import imagezmq
#import argparse
import imutils
import time
import dlib
import cv2

#ap.add_argument("-mW", "--montageW", required=True, type=int,
#	help="montage frame width")
#ap.add_argument("-mH", "--montageH", required=True, type=int,
#	help="montage frame height")

ESTIMATED_NUM_PIS = 1
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

class stream_detector:

    def __init__(
        self,
        prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
        model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
        confidence = 0.4,
        skipframes = 30
    ):

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    
        self.confidence = confidence
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        self.imageHub = imagezmq.ImageHub()
		
        self.W = None
        self.H = None
		#self.mW = args["montageW"]
		#self.mH = args["montageH"]

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		#self.frameDict = {}
        self.trackers = []
        self.trackableObjects = {}

        self.totalFrames = 0
        self.skipframes = skipframes
        self.lastActive = {}
        self.lastActiveCheck = datetime.now()

        self.fps = FPS().start()
        print("Initiate detector...")

    def main(self):
        
        totalUp = 0
        totalDown = 0

        try:
            while True:

                (rpiName, frame) = self.imageHub.recv_image()
                self.imageHub.send_reply(b'OK')
                # frame = frame[1] if args.get("input", False) else frame
                if rpiName not in self.lastActive.keys():
                    print("[INFO] receiving data from {}...".format(rpiName))
                self.lastActive[rpiName] = datetime.now()
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.W is None or self.H is None:
                    (self.H, self.W) = frame.shape[:2]

                status = "Waiting"
                rects = []

                if self.totalFrames % self.skipframes == 0:
                
                    status = "Detecting"
                    self.trackers = []

                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
                    self.net.setInput(blob)
                    detections = self.net.forward()

                    for i in np.arange(0, detections.shape[2]):
                        
                        confidence = detections[0, 0, i, 2]

                        if confidence > self.confidence:
                    
                            idx = int(detections[0, 0, i, 1])

                            if self.CLASSES[idx] != "person":
                                continue

                            box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                            (startX, startY, endX, endY) = box.astype("int")
                            rects.append((startX, startY, endX, endY))

                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            self.trackers.append(tracker)

                else:
                    for tracker in self.trackers:
                        
                        status = "Tracking"

                        tracker.update(rgb)
                        pos = tracker.get_position()

                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                        rects.append((startX, startY, endX, endY))

                cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

                objects = self.ct.update(rects)

                for (objectID, centroid) in objects.items():
                    
                    to = self.trackableObjects.get(objectID, None)

                    if to is None:
                        to = TrackableObject(objectID, centroid)

                    else:
                        
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)

                        if not to.counted:

                            if direction < 0 and centroid[1] < self.H // 2:
                                totalUp += 1
                                to.counted = True

                            elif direction > 0 and centroid[1] > self.H // 2:
                                totalDown += 1
                                to.counted = True

                    self.trackableObjects[objectID] = to

                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                info = [
                    ("Up", totalUp),
                    ("Down", totalDown),
                    ("Status", status),
                ]

                cv2.putText(frame, rpiName, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                #self.frameDict[rpiName] = frame

                #montages = build_montages(self.frameDict.values(), (self.W, self.H), (self.mW, self.mH))

                #for (i, montage) in enumerate(montages):
                #    cv2.imshow("Home pet location monitor ({})".format(i),montage)

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if (datetime.now() - self.lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
                # loop over all previously active devices
                    for (rpiName, ts) in list(self.lastActive.items()):
                # remove the RPi from the last active and frame
                # dictionaries if the device hasn't been active recently
                        if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                            print("[INFO] lost connection to {}".format(rpiName))
                            self.lastActive.pop(rpiName)
                            #frameDict.pop(rpiName)

                # set the last active check time as current time
                self.lastActiveCheck = datetime.now()

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

                # increment the total number of frames processed thus far and
                # then update the FPS counter
                self.totalFrames += 1
                self.fps.update()
        
        except:
            pass
        # stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        cv2.destroyAllWindows()

if __name__ == '__main__':
    test = stream_detector()
    test.main()