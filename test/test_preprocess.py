import unittest
import numpy as np
from scipy.io import wavfile
import pysptk
import os
from os.path import join
import sys

cwd = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(cwd)[0]
sys.path.append(os.path.join(_root, 'src'))

import espp.preprocess as prep
import espp.analysis as alysis
filename = os.path.join(cwd, "data/tmp.wav")
fs, x = wavfile.read(filename)
chunk_size = 1024

voice_samples = alysis.starting_info(
    x, alysis.pitch_detect(
        x, fs, chunk_size), fs, chunk_size)['VSamp']


class TestPreProcess(unittest.TestCase):

    def test_utterance_region_begin_samples(self):
        self.assertIsNotNone(
            prep.utterance_region_begin_samples(voice_samples))

    def test_utterance_chunk(self):
        self.assertIsNotNone(
            prep.utterance_chunk(
                voice_samples,
                prep.utterance_region_begin_samples(voice_samples)[1]))

    def test_pre_process(self):
        self.assertIsNotNone(prep.pre_process(voice_samples))


if __name__ == '__main__':
    unittest.main()
