
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

class detector:

    def __init__(
        self,
        prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
        model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
        confidence = 0.4,
        skipframes = 10
    ):

        print("Initiate detector...")

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    
        self.confidence = confidence
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        #self.vs = VideoStream(src = 0, usePiCamera = True).start()
        self.vs = VideoStream(src = 0).start()

        self.W = None
        self.H = None

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        self.totalFrames = 0
        self.skipframes = skipframes

        self.fps = FPS().start()

    def main(self):

        while True:

            frame = self.vs.read()
            frame = frame[1]
            # frame = frame[1] if args.get("input", False) else frame
            frame = imutils.resize(frame, width=250)
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


            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            self.totalFrames += 1
            self.fps.update()
        # stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        self.vs.stop()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    test = detector()
    test.main()

