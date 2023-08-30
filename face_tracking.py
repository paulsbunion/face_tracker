# import the necessary packages
# follow the tracker stuff @ https://github.com/gdiepen/face-recognition/blob/master/track%20multiple%20faces/demo%20-%20track%20multiple%20faces.py#L208
#from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import dlib
from picamera2 import Picamera2

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
def updateTrackedFaces(trackedFaces, baseImage):
	# check quality of tracked faces
	faceIdsToDelete = []
	for faceId in trackedFaces.keys():
		# update the tracker fit
		trackQuality = trackedFaces[faceId].update(baseImage);
		if (trackQuality < 7):
			faceIdsToDelete.append(faceId)
	for faceId in faceIdsToDelete:
		print("lost tracker for:", faceId)
		trackedFaces.pop(faceId, None)

'''for any found faces, check if it is a new face that needs to be tracked'''
def addNewTrackedFaces(foundFaces, trackedFaces, baseImage):
	if foundFaces is None:
		return
	
	'''match each found face with the correspondig tracker(if exists) 
	 or add a new one'''
	for key_name in foundFaces:
		_data = foundFaces[key_name]
		'''the tuple rect boxes created from face_recognition.face_locations
		are in CSS order (top right bottom left)'''
		y = int(_data[0])
		x2 = int(_data[1])
		y2 = int(_data[2])
		x = int(_data[3])
		
		# calculate the center
		x_mid = (x + x2)*0.5
		y_mid = (y + y2)*0.5
		
		# check if this point is inside any of the trackers
		matchedFaceId = None
		for tracker_key in trackedFaces:
			tracker_data = trackedFaces[tracker_key]
			tracker_position = tracker_data.get_position()
			t_x = int(tracker_position.left())
			t_y = int(tracker_position.top())
			t_w = int(tracker_position.width())
			t_h = int(tracker_position.height())
			#print("tracker data")
			#print("(", t_x, t_y, ") (", t_w, t_h, ")" )

			#calculate the centerpoint
			t_x_mid = t_x + 0.5 * t_w
			t_y_mid = t_y + 0.5 * t_h
			
			#check if the centerpoint of the face is within the 
			#rectangleof a tracker region. Also, the centerpoint
			#of the tracker region must be within the region 
			#detected as a face. If both of these conditions hold
			#we have a match
			if ( ( t_x <= x_mid   <= (t_x + t_w)) and 
				 ( t_y <= y_mid   <= (t_y + t_h)) and 
				 ( x   <= t_x_mid <= (x2)) and 
				 ( y   <= t_y_mid <= (y2))):
				matchedFaceId = tracker_key
				
		'''if no match was found, add a new tracker'''
		if matchedFaceId == None:
			print("addind new tracker for:", key_name)
			rect = dlib.rectangle(x-10, y-20, x2+10, y2+20)
			#Create and store the tracker 
			tracker = dlib.correlation_tracker()
			tracker.start_track(baseImage, rect)
			trackedFaces[key_name] = tracker

	

'''detect using opencv and track with dlib'''
def detectAndRecognizeFaces(baseImage):
	global currentname	
	
	# Detect the face boxes
	boxes = face_recognition.face_locations(baseImage)
	
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(baseImage, boxes)
	names = []
	
	# TODO: add return if boex is none or empty

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name

		# update the list of names
		names.append(name)
		#print("box 0 data: ", boxes[0][0])
		
		'''note: this tuple is stored in CSS order (top right bottom left)'''
		#return a tuple of the data
		return dict(zip(names, boxes))

def drawFaceBoundingBoxes(frame, trackedFaces):
	if trackedFaces is None:
		return
	# loop over the recognized faces
	for name in trackedFaces:
		pos = trackedFaces[name].get_position()
		top = int(pos.top())
		left = int(pos.left())
		bottom = int(pos.bottom())
		right = int(pos.right())
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)



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

