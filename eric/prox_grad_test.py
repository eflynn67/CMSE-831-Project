import numpy as np
import scipy
import h5py
import modules
import matplotlib.pyplot as plt
import time
import os
def prox_g(x,t):
    result= np.zeros(x.shape)
    for k in range(len(x)):
        if abs(x[k]) <= t :
            result[k] = 0
        else:
            result[k] = x[k] - t*np.sign(x[k])
    return result

###############################################################################

outputFile = 'test.h5'
try:
    os.remove(outputFile)
except FileNotFoundError:
    pass
uniqueCoords, zz = modules.read_pes('../data/UNEDF1.dat')


## sample the PES using the hyperbolic cross sampling
# controls how many points to include
sparseGridOrder = 6
# returns the sampled coordinates and energies at those coordinates
samplePoints, sampleEvals = modules.SamplePoints.get_sparse_grid_2d(uniqueCoords,zz,sparseGridOrder,
                                                            outputFile,flip=True)

# Now construct the Matrix C to be used in the compressed sensing algo
# arrDim = full dimension (m,n) of the pes in fourier space
#samplematrix is the mxn matrix with non-zero components correspoding to the spatial samples
arrDim, sampleMatrix = modules.SamplePoints.get_sample_matrix_2d(sparseGridOrder)
# now construct the matric representation of e_1 X e_2
dft1d = scipy.fft.fft(np.eye(arrDim))

CPsi = modules.make_CPsi(sampleMatrix,dft1d)

# warning  this one makes a huge matrix.
#Psi = np.kron(dft1d,dft1d)

## define an output file.
h5File = h5py.File(outputFile,'a')
h5File.attrs.create('basisName','dft')
h5File.close()

# lambda for the opt problem
lamb = 500

lassoClass  = modules.ComplexLasso(lamb,sampleEvals,outputFile,CPsi=CPsi)

nIters = 1000
stepSize = 10**(-4)

#opt = modules.ComplexGradientDescent(lassoClass,nIters,stepSize)
#opt = modules.ProxGradientDescent(lassoClass,nIters,stepSize)
gamma0 = 1
m = 1
opt = modules.AccProxGradientDescent(lassoClass,nIters,stepSize,m,gamma0)
s = np.zeros(sampleMatrix.shape[1]).astype(complex)
print(s)
rt0 = time.time()
s = opt.run(s,outputFile)
rt1 = time.time()
print("Run time: %.6f"%(rt1-rt0))

###############################################################################
# Plot the results and stuff

print("Run time: %.6f"%(rt1-rt0))

fig, ax = plt.subplots()
ax.plot(opt.lossVals)
ax.set(yscale='log')
ax.set(title='Loss vs iteration')


fig, ax = plt.subplots()
ax.plot(np.real(s))
ax.plot(np.imag(s))
ax.set(title='real and imaginary part of s')
#%%
# takes long time. matrix is large
#t0 = time.time()
#newPes = Psi @ s
#t1 = time.time()
#print('Matrix multiplication time: %.6f'%(t1-t0))
# newPes = np.fft.fftshift(np.fft.fft(s))

t0 = time.time()
newPes = scipy.fft.fft2(s.reshape(2*(arrDim,)))
t1 = time.time()
print('FFT time: %.6f'%(t1-t0))

rmse, newCoordVals, pesDiff = modules.pes_rmse(uniqueCoords,zz,newPes.reshape(2*(arrDim,)),flip=True)

h5File = h5py.File(outputFile,'a')
h5File.attrs.create('pesRMSE',rmse)
h5File.close()

#%%
fig, ax = plt.subplots()
cf = ax.contourf(*newCoordVals,pesDiff.T.clip(-5,5),cmap='Spectral_r',levels=30)
plt.colorbar(cf,ax=ax)
ax.set(title='PES Difference')

fig, ax = plt.subplots()
cf = ax.contourf(np.real(newPes).reshape((arrDim,arrDim)).T.clip(-30,30),
                  cmap='Spectral_r',levels=30)
# cf = ax.pcolormesh(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).clip(-30,30),
#                    cmap='Spectral_r')
plt.colorbar(cf,ax=ax)
ax.set(title='Fit PES Real Component')

fig, ax = plt.subplots()
cf = ax.contourf(np.imag(newPes).reshape((arrDim,arrDim)).T.clip(-1,1),
                 cmap='Spectral_r',levels=30)
plt.colorbar(cf,ax=ax)
ax.set(title='Fit PES Imaginary Component')

#%%
sReshaped = s.reshape(2*(arrDim,))
fig, ax = plt.subplots()
cf = ax.pcolormesh(np.fft.fftshift(np.real(sReshaped)),cmap='binary')
plt.colorbar(cf,ax=ax)
plt.title('real part of surface')
fig, ax = plt.subplots()
cf = ax.pcolormesh(np.fft.fftshift(np.imag(sReshaped)),cmap='binary')
plt.colorbar(cf,ax=ax)
plt.title('imaginary part of surface')


