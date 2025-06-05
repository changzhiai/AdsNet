#!/opt/anaconda3/bin/python
from ase.io import read
import sys

# name = sys.argv[1]
name = '/Users/changzhi/Library/CloudStorage/OneDrive-SLACNationalAcceleratorLaboratory/SLAC/AdsNet/test/painn/Pt_ads.traj'

images = read(name, ':')
print(len(images))
