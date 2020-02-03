# demon demo

# still need to write fuzzy segmentation scripts in python
# for now using segmented output via Matlab

from ComputeDeformation import ComputeDeformation
from scipy.io import loadmat

matdata = loadmat('defuzzed.mat')
L1 = matdata['L1']
L2 = matdata['L2']

# define parameters
MaxIter = 300
NumPyramids = 4
filterSize = [200, 200]
Tolerance = 0
alpha = 0.45
plotFreq = 0
filterSigma = 30

# try to run the deformation
Tx, Ty = ComputeDeformation(L1,L2,MaxIter,NumPyramids,
                           filterSize,filterSigma,
                           Tolerance,alpha,plotFreq)
