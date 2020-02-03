import numpy as np
from scipy.interpolate import interp2d

def DeformImage(StartingImage, Tx, Ty):
    """
    DeformImage - Compute deformed image,
                  using the transformation matrices (Tx,Ty)
    - Original Matlab implementation by Lewis Li (lewisli@stanford.edu)
    - Converted into Python by J. Hariharan (jayaram.hariharan@utexas.edu)
      Feb 1 2020
    """
    # create interpolation grid
    lilx = np.arange(0,np.shape(StartingImage)[0],1)
    lily = np.arange(0,np.shape(StartingImage)[1],1)
    [X,Y] = np.meshgrid(lilx, lily)

    # transformed coordinates
    TXNew = X.T + Tx
    TYNew = Y.T + Ty

    # clamp ends
    TXNew[TXNew<0] = 0
    TYNew[TYNew<0] = 0
    #TXNew[TXNew>lilx] = lilx
    #TYNew[TYNew>lily] = lily

    # interpolate
    Outputinterp = interp2d(lilx,lily,StartingImage.T,kind='cubic')
    OutputImage = Outputinterp(lily,lilx)

    return OutputImage
