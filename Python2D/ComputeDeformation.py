import sys
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import correlate
import cv2
from DeformImage import DeformImage

def fspecial(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    From: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    From: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def ComputeDeformation(I1,I2,MaxIter,NumPyramids,
                       filterSize,filterSigma,Tolerance,
                       alpha,plotFreq):
    """
    Demon Algorithm
    - Implementation of Thirion's Demon Algorithm in 2D.
      Computes the deformation between I2 and I1.
    - Original Matlab implementation by Lewis Li (lewisli@stanford.edu)
    - Converted to Python by J. Hariharan (jayaram.hariharan@utexas.edu)
      Feb 1 2020

    Function Parameters:
        - I1: Image 1
        - I2: Image 2
        - MaxIter: Maximum of iterations
        - NumPyramids: Number of pyramids for downsizing
        - filterSize: Size of Gaussian filter used for smoothing deformation meshgrid
        - Tolerance: MSE convergence tolerance
        - alpha: Constant for Extended Demon Force (Cachier 1999). If alpha==0, run regular Demon force
        - plotFreq: Number of iterations per plot update. Set to 0 to turn off plotting
    """

    # How much MSE can increase between iterations before terminating
    MSETolerance = 1 + Tolerance
    MSEConvergenceCriterion = Tolerance

    # alpha (noise) constant for Extended Demon Force
    alphaInit = alpha

    # pyramid step size
    pyStepSize = 1.0 / NumPyramids

    # initial transformation field is smallest possible size
    initialScale = 1 * pyStepSize
    prevScale = initialScale

    TxPrev = np.zeros( [int(np.ceil(np.shape(I1)[0]*initialScale)) , int(np.ceil(np.shape(I1)[1]*initialScale)) ] )
    TyPrev = np.zeros( [int(np.ceil(np.shape(I1)[0]*initialScale)) , int(np.ceil(np.shape(I1)[1]*initialScale)) ] )

    # iterate over each pyramid
    for pyNum in range(1,NumPyramids+1):
        print('pyNum: ' + str(pyNum))

        scaleFactor = pyNum * pyStepSize

        # increase size of smoothing filter
        Hsmooth = fspecial([filterSize[0]*scaleFactor,filterSize[1]*scaleFactor],filterSigma*scaleFactor)

        # only needed if using extended demon force
        alpha = alphaInit / pyNum

        # resize images according to pyramid steps
        ### S and M definition currently not same as MATLAB version !!!
        I2 = resize(I2, [np.ceil(np.shape(I2)[0]*scaleFactor),np.ceil(np.shape(I2)[1]*scaleFactor)])
        I2 = np.double(I2)
        out2 = np.zeros(I2.shape, np.double)
        S = cv2.normalize(I2, out2, 1.0, 0.0, cv2.NORM_MINMAX)

        I1 = resize(I1, [np.ceil(np.shape(I1)[0]*scaleFactor),np.ceil(np.shape(I1)[1]*scaleFactor)])
        I1 = np.double(I1)
        out1 = np.zeros(I1.shape, np.double)
        M = cv2.normalize(I1, out1, 1.0, 0.0, cv2.NORM_MINMAX)

        # compute MSE
        prevMSE = np.abs(M-S)**2
        StartingImage = M

        # histogram match
        # M = hist_match(M, S)

        # transformation fields:
        # transformation field for current pyramid is the transformation field
        # at the previous pyramid bilinearly interpolated to current size. The
        # magnitudes of the deformations needs to be scaled by the change in
        # scale too.
        Tx = resize(TxPrev, np.shape(S))*scaleFactor/prevScale
        Ty = resize(TyPrev, np.shape(S))*scaleFactor/prevScale

        M = DeformImage(StartingImage,Tx,Ty)
        prevScale = scaleFactor

        [Sy,Sx] = np.gradient(S)

        for itt in range(1,MaxIter+1):
            print('itt: ' + str(itt))

            # difference image between moving and static image
            import pdb; pdb.set_trace()
            Idiff = M - S

            if alpha == 0:
                # default demon force (Thirion 1998)
                Ux = -(Idiff*Sx) / ( (Sx**2 + Sy**2) + Idiff**2 )
                Uy = -(Idiff*Sy) / ( (Sx**2 + Sy**2) + Idiff**2 )
            else:
                # extended demon force. faster convergence but more unstable
                # (Cachier 1999, He Wang 2005)
                [My,Mx] = np.gradient(M)
                Ux = -Idiff * ( (Sx/((Sx**2+Sy**2) + alpha**2 * Idiff**2) ) + (Mx/((Mx**2+My**2)+alpha**2 * Idiff**2)) )
                Uy = -Idiff * ((Sy/((Sx**2+Sy**2)+alpha**2*Idiff**2))+(My/((Mx**2+My**2)+alpha**2*Idiff**2)))

            # when divided by zero
            Ux[np.isnan(Ux)] = 0
            Uy[np.isnan(Uy)] = 0

            # smooth the transformation field
            Uxs = 3*correlate(Ux,Hsmooth)
            Uys = 3*correlate(Uy,Hsmooth)

            # add new transformation field to the total transformation field
            Tx = Tx + Uxs
            Ty = Ty + Uys

            M = DeformImage(StartingImage,Tx,Ty)

            D = np.abs(M-S)**2
            [sA,sB] = np.shape(S)
            MSE = np.sum( D[:]/(sA*sB) )

            if MSETolerance > 0:
                # break if MSE is increasing
                if MSE > np.sum(prevMSE)*MSETolerance:
                    print('Pyramid Level: ' + str(pyNum) + ' converged after ' + str(itt) + ' iterations.')
                    sys.exit()
                else:
                    pass

                # break if MSE isn't really decreasing much
                if np.abs(np.sum(prevMSE)-MSE)/MSE < MSEConvergenceCriterion:
                    print('Pyramid Level: ' + str(pyNum) + ' converged after ' + str(itt) + ' iterations.')
                    sys.exit()
                else:
                    pass

            else:
                pass

            # update MSE
            prevMSE = MSE

            if plotFreq > 0:
                # add plot stuff here
                pass
            else:
                pass

        # propogate transformation to next pyramid
        TxPrev = Tx
        TyPrev = Ty

    return Tx, Ty
