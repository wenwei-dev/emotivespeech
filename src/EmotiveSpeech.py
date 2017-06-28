import numpy as np
import preprocess as prep
import warnings
warnings.filterwarnings('ignore')
import batchprocess as bp
import synthesis as syn
import os

CHUNK_SIZE = 1024
FILE_NAME_PATH = "/home/dereje/Desktop/TestFolder/Test.wav"
def emotive_speech(x,fs,typeOfEmotion):
	"""
	emotive_speech(x,fs,typeOfEmotion)
			A Caller Module
		Parameter:x
				  fs
				  typeOfEmotion
		Returns: output
	"""
	TIME_STAMPS = bp.process_variables(x,fs,CHUNK_SIZE)[0]
	CONSECUTIVE_BLOCKS = bp.process_variables(x,fs,CHUNK_SIZE)[1]
	fundamental_frequency_in_blocks = bp.batch_analysis(x,fs,CHUNK_SIZE)[0]
	voiced_samples = bp.batch_analysis(x,fs,CHUNK_SIZE)[1]
	rms = bp.batch_analysis(x,fs,CHUNK_SIZE)[2]
	selected_inflect_block = bp.batch_preprocess(fundamental_frequency_in_blocks,voiced_samples,rms)
	output = bp.batch_synthesis(fs,CONSECUTIVE_BLOCKS,TIME_STAMPS,selected_inflect_block,typeOfEmotion)
	return output

if __name__ == '__main__':
	fs,x = prep.wave_file_read(FILE_NAME_PATH)
	emotive_speech(x,fs,"happy")
	emotive_speech(x,fs,"happy_tensed")
	emotive_speech(x,fs,"afraid")
	emotive_speech(x,fs,"sad")
