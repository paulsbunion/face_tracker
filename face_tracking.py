# import the necessary packages
# follow the tracker stuff @ https://github.com/gdiepen/face-recognition/blob/master/track%20multiple%20faces/demo%20-%20track%20multiple%20faces.py#L208
#from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import time
import cv2
from picamera2 import Picamera2
from face_tracking_utils import addNewTrackedFaces, detectAndRecognizeFaces, drawFaceBoundingBoxes, updateTrackedFaces

picam2 = Picamera2()
dispW=1280
dispH=720
picam2.preview_configuration.main.size = (dispW,dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
myFps=0
pos=(30,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(0,0,255)

frameCounter = -1;
displayFps = True
displayPreview = True
checkForFacesEveryXFrames = 10

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"

'''Update the list of tracked faces by checking if they still are valid
positions from the previous frame'''
'''for any found faces, check if it is a new face that needs to be tracked'''
'''detect using opencv and track with dlib'''
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# start the FPS counter
fps = FPS().start()

trackedFaces = {}

# loop over frames from the video file stream
while True:
	tStart=time.time()
	frameCounter += 1
	
	# grab the frame from the threaded video stream and resize it
	frame = picam2.capture_array()
	frame = imutils.resize(frame, width=500)
	baseImage = frame.copy()
	
	# update trackedFaces data
	updateTrackedFaces(trackedFaces, baseImage)
	
	# run the face detection and recognition every 10 frames
	# to look for new faces
	if frameCounter % checkForFacesEveryXFrames == 0:
		foundFaces = detectAndRecognizeFaces(baseImage)
		addNewTrackedFaces(foundFaces, trackedFaces, baseImage)
		
	if len(trackedFaces) > 0:
		drawFaceBoundingBoxes(frame, trackedFaces)

	# display fps
	if displayFps:
		cv2.putText(frame,str(int(myFps))+' FPS',pos,font,height,myColor,weight)
	# draw preview
	if displayPreview:
		cv2.imshow("Facial Recognition is Running", frame)
		
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
	tEnd=time.time()
	loopTime=tEnd-tStart
	myFps=.9*myFps + .1*(1/loopTime)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
picam2.stop()

