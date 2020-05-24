# RaspberryPi_PedestianCounter_Final
Release note of Pi_PedestianCounter_Project

## Hardware

* Raspberry Pi 3 model B+
* Raspberry Pi camera V2
* ethernet cable (optional: for faster and more reliable connection between client(pi) and server(PC))
* Acrylic case (additional hardware design need for real world usage)

## Software

### Dependencies

* python 3.6.7
* opencv-python 4.1.0.25
* opencv-contrib-python 4.1.1.26
* dlib 19.9.0
* imutils 0.5.2
* numpy 1.17.4
* pyzmq 18.0.1
* imagezmq
  
### Testing Result
#### Speed
Without ethernet connection: 15-18 FPS (upto wireless connection at the moment) <br>
With ethernet connection: ~25 FPS (Stable due to being direct connection and doesn't involve internet connection)
<br>
#### Accuracy
In perfect condition detection accuracy can get upto 90% but can change drasticly in extreme lighting condition. In many test, accuracy go really low as a contrast between bright area and dark area captured by the camera make an image almost indistinguishable even to human eye. 
In other test, using the system in low light condition causing a image capture to be really dark and almost indistinguishable too. 
For this particular factor, camera limitation might play a role in this poor accuracy. 
More camera configuration or hardware change might benefit if you want to use this solution in environment with poor lighting. 
Another factor that might affect detection accuracy is camera orientation. 
In one test, the camera is set to capture the bird-eye view, in this test the software failed to detect person in the image which might be because lack of training data in that particular orientation. 
In summary, this solution perform well in situation with perfect lighting environment (no extreme lighting with few variable in lighting) and camera orientation can be change according to accuracy at runtime.

Speed also play a huge role in tracking accuracy too. 
In a pervious solution with poor speed, it is found that even with test that have high detection accuracy the overall tracking result is poor as there is a jump of tracking between two sepereate object (people). 
This can be expected as the underlying tracking algorithm is centroid tracking which mark two object which have closest centroid in two frame as same object for tracking.
With poor speed this can result in a jumping track between two seperate object as a real object is farther to original position than other object which may come and take that position.
For this particular case there are two ways to increase accuracy
* Increase speed and FPS which is a way used in this solution (by using concept of transferring image to process in a hardware with more computing power) 
* Change underlying tracking algorithm to different algorithm which is more resilent to poor FPS. 

### Future plan
As I discuss previously, there are many improving opportunity for this solution which can be listed below
* Hardware especially case which is needed for real implementation of the solution.
* A way to secure reliable connection between Pi and server without ethernet connection
* A solution for improve lighting limitation such as better image processing algorithm or hard ware configuration.
* An orientation of camera which provide best perfomance (as of right now it is test and adjust which is not ideal in implementation)
* A way to make solution less susceptible to camera orientation such as algorithm change, transfer learning in image with particular orientation.
* Better tracking algorithm which is less susceptible to poor FPS.
What listed above is only an example of what can be done to a improve a solution only. 
There are plenty more of thing you can do to improve this solution such as using neural computing stick to improve computing power so that there is less needed for network transfer which may improve speed significantly too.

### Original Code

* OpenCV People Counter by Adrian Rosebrock (https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)
* Live video streaming over network with OpenCV and ImageZMQ by Adrian Rosebrock (https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/)
