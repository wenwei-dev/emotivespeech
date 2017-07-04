import numpy as np
import preprocess as prep
import warnings
warnings.filterwarnings('ignore')
import batchprocess as bp
import synthesis as syn
import os
import sys

def emotive_speech(file_name_path, chunk_size, typeOfEmotion):
    """
    emotive_speech(x,fs,typeOfEmotion)
    A Caller Module
    Parameter:  x
                fs
                typeOfEmotion
    Returns: output
    """
    fs, x = prep.wave_file_read(file_name_path)
    time_stamps = bp.process_variables(x, fs, chunk_size)[0]
    consecutive_blocks = bp.process_variables(x, fs, chunk_size)[1]
    fundamental_frequency_in_blocks = bp.batch_analysis(x, fs, chunk_size)[0]
    voiced_samples = bp.batch_analysis(x, fs, chunk_size)[1]
    rms = bp.batch_analysis(x, fs, chunk_size)[2]
    selected_inflect_block = bp.batch_preprocess(
        fundamental_frequency_in_blocks, voiced_samples, rms)
    output = bp.batch_synthesis(
        fs,
        consecutive_blocks,
        time_stamps,
        selected_inflect_block,
        typeOfEmotion)
    return output

if __name__ == '__main__':
    file_name_path = sys.argv[1]
    chunk_size = int(sys.argv[2])
    typeOfEmotion = sys.argv[3]
    emotive_speech(file_name_path, chunk_size, typeOfEmotion)
