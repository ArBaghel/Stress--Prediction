import unittest
import numpy as np
from apptest1 import bandpass_filter, pan_tompkins_rpeak

class TestApp(unittest.TestCase):
    def test_bandpass_filter(self):
        # Generate a dummy 700 Hz signal (e.g. 1 second of noise + sine wave)
        fs = 700
        t = np.linspace(0, 1, fs, endpoint=False)
        # 2 Hz sine wave (should pass) + 100 Hz high frequency noise (should be filtered)
        clean_signal = np.sin(2 * np.pi * 2 * t)
        noisy_signal = clean_signal + 0.5 * np.sin(2 * np.pi * 100 * t)
        
        filtered = bandpass_filter(noisy_signal)
        
        self.assertEqual(len(filtered), fs)
        # The variance of high-frequency noise should be significantly reduced
        self.assertLess(np.std(filtered - clean_signal), np.std(noisy_signal - clean_signal))

    def test_pan_tompkins_rpeak(self):
        # Test with a dummy flat signal (should return no peaks)
        dummy_signal = np.zeros(1400)
        peaks = pan_tompkins_rpeak(dummy_signal, fs=700)
        self.assertEqual(len(peaks), 0)

if __name__ == '__main__':
    unittest.main()
