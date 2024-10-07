from queue import Queue

WITH_SOUND = 0  # for checking sound
WITH_AUDIO = 0  # check if there are audio embedding
CAPTURE_FRAME = None  # checking the first lunching
VIDEO_File_NAME = "D:/Research/code/MSA_Vis/data/videos/"  # saving videos with sound

V_QUEUE = Queue()  # used for video data process
A_QUEUE = Queue()  # used for audio data process

A_FEATURE = None
T_FEATURE = None
