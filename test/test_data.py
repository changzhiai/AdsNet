from adsnet.data import AseData
from ase.io import read
import numpy as np
import unittest,os

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class AseDataTestCase(unittest.TestCase):

    # def test_AseData(self):
    #     images = read(f'{path}/Pt_ads.traj', ':')
    #     asedata = AseData(images, cutoff=5.0)
    #     assert isinstance(asedata[0], dict)

    #     print(f'asedata[0]: {asedata[0]}')
    #     for i in range(len(images)):
    #         print(asedata[i].keys(), )

    def test_AseData_forces(self):
        images = read(f'{path}/test.traj', ':')
        asedata = AseData(images, cutoff=5.0)
        assert isinstance(asedata[0], dict)

        # print(f'asedata[0]: {asedata[0]}')
        for i in range(len(images)):
            ads_idx = asedata[i]['ads_idx']
            forces = images[i].get_forces()
            print(asedata[i].keys(), )
            print(ads_idx[0])
            print(asedata[i]['delta_forces'][ads_idx[0]])

            print('adsorbate forces difference: ', (forces-asedata[i]['delta_forces'].cpu().numpy())[ads_idx[0]])
            
            assert np.allclose(forces[ads_idx[0]], asedata[i]['delta_forces'][ads_idx[0]].cpu().numpy(), \
                                rtol=1e-4, atol=10-4), 'adsorbate forces not match! '


if __name__ == '__main__':
    unittest.main()

