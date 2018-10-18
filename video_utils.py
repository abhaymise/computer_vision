
# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time
 

# import the necessary packages

from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


import file_system_utils as fs_utls

import sys
sys.path.insert(0,'/home/ubuntu/Documents/official_projects/nsfw/content-based-video-recommendation/c3d')

import warnings
from c3d import C3D
from sports1M_utils import preprocess_input
from keras.models import Model


# import the necessary packages
from threading import Thread
import sys
import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

class FileVideoStream:
	def __init__(self, path, transform=None, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		
		self.stopped = False
		self.transform = transform

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)
	
	@property
	def get_video_stats(self):

		cap = self.stream
		info = {
        "framecount": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    	}
		return info

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return

				# if there are transforms to be done, might as well
				# do them on producer thread before handing back to
				# consumer thread. ie. Usually the producer is so far
				# ahead of consumer that we have time to spare.
				#
				# Python is not parallel but the transform operations
				# are usually OpenCV native so release the GIL.
				#
				# Really just trying to avoid spinning up additional
				# native threads and overheads of additional 
				# producer/consumer queues since this one was generally
				# idle grabbing frames.
				if self.transform:
					frame = self.transform(frame)

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue  

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream. 
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

def read_vid_as_array(vid_path,show_frame=False):
    """
    Reads video faster and returns the read array.
    """
    print("[INFO] starting video file thread...")
    
    fvs = FileVideoStream(vid_path , ).start()

    print(fvs.get_video_stats)
    time.sleep(1.0)

    # start the FPS timer
    fps = FPS().start()

    all_frames = []
    # loop over frames from the video file stream
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()
        # frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = np.dstack([frame, frame, frame])
        # c3frame = np.expand_dims(frame,0)
        # c3d_model.extract_features(c3frame)
        all_frames.append(frame)

        if show_frame :
            # display the size of the queue on the frame
            cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

            # show the frame and update the FPS counter
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

        fps.update()

       # do a bit of cleanup
    # cv2.destroyAllWindows()
    fvs.stop()
     # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    vid_arry = np.array(all_frames)
    return vid_arry
    
    # print(c3d_model.get_3d_feature(vid_arry).shape)


class Extract_c3d():
    def __init__(self):
        base_model = C3D(weights='sports1M')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)
        print("C3D model loaded ... ")
        
    def __get_features(self,vid):
        if len(vid)<=16:
            warnings.warn("The array size is less than 16 ...")
        x = preprocess_input(vid)
        features = self.model.predict(x)
        return features

    def __batch_idx(self,video_len,batch_size):
        for idx in range(0,video_len,batch_size):
            yield range(idx,min(idx+batch_size,video_len))
            
    def get_3d_feature(self,vid_array,batch_size=16):
       
        bat = self.__batch_idx(vid_array.shape[0],batch_size)
        batch_indexes = list(bat)
        print('total temporal vectors ',len(batch_indexes))
        arr = np.empty((0,4096))
        for val in batch_indexes:
            arr=np.append(arr,self.__get_features(vid_array[val]),axis=0) 
        return arr

from pathlib import Path

def get_encoding_path_from_id(wid,dest_path):
    """
    dest_path / vid_id / <data.npy>
    """
    file_path = Path(dest_path) / str(wid) / 'data.npy'
    return str(file_path)

def get_encoding_path_from_source_loc(source_file_path,dest_path):
    """
    dest_path / vid_id / <data.npy>
    """
    file_path = Path(dest_path) / str(fs_utls.get_stem(source_file_path)) / 'data.npy'
    return str(file_path)

def save_encoding(datapy,file_path,overwrite=False):
    fs_utls.create_subdirs(file_path,overwrite=overwrite)
    print("saving encodings ... ")
    np.save(file_path, datapy)
    return None

def load_encoding(file_path):
    if fs_utls.if_path_exists(file_path):
        print("loading encodings ... ")
        return np.load(file_path)
    else : 
        print ("Can't load file .... ")
        return None

def c3d_feature_generator(c3d_model,video_list):
    for vid_path in video_list:
        vid_array = read_vid_as_array(vid_path)
        c3d_array = c3d_model.get_3d_feature(vid_array)
        yield c3d_array , vid_path

def main():

    ## construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    args = vars(ap.parse_args())

    ## load c3d model
    c3d_model = Extract_c3d()

    vid_list = [args['video']]

    c3d_project = ''

    encoding_dest_dir = 'abhay'

    for encoding , path in c3d_feature_generator(c3d_model,vid_list):
        file_path = get_encoding_path_from_source_loc(path,encoding_dest_dir)
        # print(file_path)
        print(save_encoding(encoding,file_path,True))

if __name__ == '__main__':
# get_3d_feature
    main()
    