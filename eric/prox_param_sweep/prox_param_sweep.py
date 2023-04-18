import numpy as np
import scipy
import h5py
import time
import os
import sys
import modules
def prox_g(x,t):
    result= np.zeros(x.shape)
    for k in range(len(x)):
        if abs(x[k]) <= t :
            result[k] = 0
        else:
            result[k] = x[k] - t*np.sign(x[k])
    return result

###############################################################################



uniqueCoords, zz = modules.read_pes('../../data/UNEDF1.dat')


## sample the PES using the hyperbolic cross sampling
# controls how many points to include
sparseGridOrder = 6
# returns the sampled coordinates and energies at those coordinates


# Now construct the Matrix C to be used in the compressed sensing algo
# arrDim = full dimension (m,n) of the pes in fourier space
#samplematrix is the mxn matrix with non-zero components correspoding to the spatial samples
arrDim, sampleMatrix = modules.SamplePoints.get_sample_matrix_2d(sparseGridOrder)
# now construct the matric representation of e_1 X e_2
dft1d = scipy.fft.fft(np.eye(arrDim))

CPsi = modules.make_CPsi(sampleMatrix,dft1d)

# warning  this one makes a huge matrix.
#Psi = np.kron(dft1d,dft1d)





nIters = 1000

# lambda for the opt problem
lambArr = [50,100,500,1000,5000,10000]
stepSizeArr = [10**(-8), 10**(-7), 10**(-6), 10**(-5), 10**(-4)]
gamma0Arr = [.1,.5,1,5,10]
mArr = [.1,.5,1,5,10]

for lamb in lambArr:
    for stepSize in stepSizeArr:
        for gamma0 in gamma0Arr:
            for m in mArr:
                outputFile = f'AccProx_lamb_{lamb}_step_{stepSize}_gamma0_{gamma0}_m_{m}.h5'
                try:
                    os.remove(outputFile)
                except FileNotFoundError:
                    pass
                
                samplePoints, sampleEvals = modules.SamplePoints.get_sparse_grid_2d(uniqueCoords,zz,sparseGridOrder,
                                                                            outputFile,flip=True)
                ## define an output file.
                h5File = h5py.File(outputFile,'a')
                h5File.attrs.create('basisName','dft')
                h5File.close()
                
                lassoClass  = modules.ComplexLasso(lamb,sampleEvals,outputFile,CPsi=CPsi)
                opt = modules.AccProxGradientDescent(lassoClass,nIters,stepSize,m,gamma0)
                s = np.zeros(sampleMatrix.shape[1]).astype(complex)
                rt0 = time.time()
                s = opt.run(s,outputFile,verbose = False)
                rt1 = time.time()
                print("Run time: %.6f"%(rt1-rt0))
                print(50*'=')
                print(f'Completed: lamb = {lamb}, stepSize = {stepSize} , gamma0 = {gamma0}, m = {m}')
                ###############################################################################
                # Plot the results and stuff
                
               