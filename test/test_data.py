from adsnet.data import AseData
from ase.io import read
import unittest,os

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class AseDataTestCase(unittest.TestCase):

    def test_AseData(self):
        images = read(f'{path}/Pt_ads.traj', ':')
        asedata = AseData(images, cutoff=5.0)
        assert isinstance(asedata[0], dict)

        print(f'asedata[0]: {asedata[0]}')

if __name__ == '__main__':
    unittest.main()

