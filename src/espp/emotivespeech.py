import numpy as np
import preprocess as prep
import batchprocess as bp
import synthesis as syn
import os
import sys


def emotive_speech(fname, chunk_size, typeOfEmotion):
    """
    emotive_speech(x,fs,typeOfEmotion)
    A Caller Module
    Parameter:  x
                fs
                typeOfEmotion
    Returns: output
    """
    fs, x = prep.wave_file_read(fname)
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
    ofile = '{}/out.wav'.format(os.path.dirname(fname))
    output.build(fname, ofile)
    return output

if __name__ == '__main__':
    fname = sys.argv[1]
    chunk_size = int(sys.argv[2])
    typeOfEmotion = sys.argv[3]
    emotive_speech(fname, chunk_size, typeOfEmotion)
