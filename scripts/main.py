import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import h5py
import pandas as pd
import sys, os
import time

from scipy.interpolate import RectBivariateSpline

def read_pes(fName,returnFormat='array'):
    pes = pd.read_csv('../data/'+fName,sep='\t')
    pes['EHFB'] -= pes['EHFB'].mean()
    uniqueCoords = [np.unique(pes[col]) for col in ['Q20','Q30']]
    zz = pes['EHFB'].to_numpy().reshape([len(u) for u in uniqueCoords])
    
    if returnFormat == 'array':
        return uniqueCoords, zz
    elif returnFormat == 'dataframe':
        return pes
    
def subset_sum(numbers, target, partial=[], partial_sum=0):
    #From https://stackoverflow.com/a/4633515
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)
    
def _flip_pes(uniqueCoords,zz):
    zz = np.vstack((np.flip(zz[1:],axis=0),zz))
    zz = np.hstack((np.flip(zz[:,1:],axis=1),zz))
    
    xFlipped = np.concatenate((-np.flip(uniqueCoords[0][1:]),uniqueCoords[0]))
    yFlipped = np.concatenate((-np.flip(uniqueCoords[1][1:]),uniqueCoords[1]))
    return xFlipped, yFlipped, zz
    
class SamplePoints:
    @staticmethod
    def sparse_1d(n):
        arr = np.arange(0,2**n)
        return 2**(-n)*arr
    
    @staticmethod
    def sparse_2d(n):
        sparseArrList = []
        
        listOf1dVals = [SamplePoints.sparse_1d(i) for i in range(n+1)]
        
        listOfCombinations = [l for l in list(subset_sum(range(n+1),n)) if len(l)==2]
        
        for l in listOfCombinations:
            sparseArrList.append(list(itertools.product(listOf1dVals[l[0]],listOf1dVals[l[1]])))
            sparseArrList.append(list(itertools.product(listOf1dVals[l[1]],listOf1dVals[l[0]])))
        
        arrOut = np.array(sparseArrList).reshape((-1,2))
        
        return np.unique(arrOut,axis=0)
    
    @staticmethod
    def get_sparse_grid_2d(uniqueCoords,zz,sparseGridOrder,fName,flip=False):
        sparse2d = SamplePoints.sparse_2d(sparseGridOrder)
        
        if flip:
            xFlipped, yFlipped, zz = _flip_pes(uniqueCoords,zz)
            
            pes_interp = RectBivariateSpline(xFlipped,yFlipped,zz)
            
            sparse2d[:,0] = sparse2d[:,0]*(xFlipped[-1] - xFlipped[0]) + xFlipped[0]
            sparse2d[:,1] = sparse2d[:,1]*(yFlipped[-1] - yFlipped[0]) + yFlipped[0]
        else:
            pes_interp = RectBivariateSpline(*uniqueCoords,zz)
            sparse2d[:,0] *= uniqueCoords[0][-1]
            sparse2d[:,1] *= uniqueCoords[1][-1]
            
        h5File = h5py.File(fName,'a')
        h5File.attrs.create('samplingMethod','sparse2d')
        h5File.create_group('samplingParams')
        h5File['samplingParams'].attrs.create('order',sparseGridOrder)
        h5File['samplingParams'].attrs.create('flip',flip)
        h5File.close()
            
        return sparse2d, pes_interp(*sparse2d.T,grid=False)
    
    @staticmethod
    def get_sample_matrix_2d(sparseGridOrder):
        sparse2d = SamplePoints.sparse_2d(sparseGridOrder)
        
        fullArrSize1d = len(sparse2d[sparse2d[:,0]==0]) #Number of elements along one axis
        
        sampleMatrix = np.zeros((len(sparse2d),fullArrSize1d,fullArrSize1d))
        sparseGridIndices = (sparse2d * 2**sparseGridOrder).astype(int)
        for (rowIter,row) in enumerate(sparseGridIndices):
            sampleMatrix[(rowIter,)+tuple(row)] = 1
        sampleMatrix = sampleMatrix.reshape((len(sparse2d),-1))
        
        return fullArrSize1d, sampleMatrix
    
def pes_rmse(uniqueCoords,zz,newPes,flip=False):
    if flip:
        xFlipped, yFlipped, zz = _flip_pes(uniqueCoords,zz)
        pes_interp = RectBivariateSpline(xFlipped,yFlipped,zz)
        
        xMin, xMax = xFlipped[0], xFlipped[-1]
        yMin, yMax = yFlipped[0], yFlipped[-1]
    else:
        pes_interp = RectBivariateSpline(*uniqueCoords,zz)
        
        xMin, xMax = uniqueCoords[0][0], uniqueCoords[0][-1]
        yMin, yMax = uniqueCoords[1][0], uniqueCoords[1][-1]
        
    x = np.linspace(xMin,xMax,newPes.shape[0])
    y = np.linspace(yMin,yMax,newPes.shape[0])
    
    evalLocs = np.array(list(itertools.product(x,y)))
    
    originalPes = pes_interp(*evalLocs.T,grid=False)
    
    pesDiff = originalPes.reshape(newPes.shape)-newPes
    rmse = np.linalg.norm(pesDiff)/np.sqrt(newPes.size)
    return rmse, (x,y), pesDiff

class ComplexLasso:
    def __init__(self,lamb,sampleMatrix,Psi,y,fileName):
        self.lamb = lamb
        self.sampleMatrix = sampleMatrix
        self.Psi = Psi
        self.y = y
        
        self.fileName = fileName
        
        self._setup_matrix_products()
        
    def _setup_matrix_products(self):
        #Can be moved elsewhere later
        C_Psi = self.sampleMatrix @ self.Psi
        self.PsiDagger_CT_C_Psi = np.conjugate(C_Psi).T @ C_Psi
        self.yT_C_A = self.y.T @ np.real(C_Psi)
        self.yT_C_B = self.y.T @ np.imag(C_Psi)
        self.yT_y = self.y.T @ self.y
        return None
        
    def loss(self,s):
        a = np.real(s)
        b = np.imag(s)
        
        f1 = a.T @ self.PsiDagger_CT_C_Psi @ a + b.T @ self.PsiDagger_CT_C_Psi @ b + self.yT_y
        f1 += -2*(self.yT_C_A @ a - self.yT_C_B @ b)
        
        f2 = self.lamb * np.sum(np.sqrt(a**2+b**2))
        return f1 + f2
    
    def _grad_f1(self,a,b):
        return 2*(self.PsiDagger_CT_C_Psi @ a - self.yT_C_A), 2*(self.PsiDagger_CT_C_Psi @ b + self.yT_C_B)
    
    def _grad_f2(self,a,b):
        return self.lamb*a/np.sqrt(a**2 + b**2), self.lamb*b/np.sqrt(a**2 + b**2)
    
    def grad(self,s):
        a = np.real(s)
        b = np.imag(s)
        
        f1_a, f1_b = self._grad_f1(a,b)
        f2_a, f2_b = self._grad_f2(a,b)
        
        return np.real(f1_a+f2_a), np.real(f1_b+f2_b)
    
    def write_statistics(self):
        h5File = h5py.File(self.fileName,'a')
        
        h5File.attrs.create('lossClass','ComplexLasso')
        h5File.create_group('lossParams')
        h5File['lossParams'].attrs.create('lambda',self.lamb)
        
        h5File.close()
        return None
    
class ComplexGradientDescent:
    def __init__(self,lossClass,maxIters,stepSize):
        self.lossClass = lossClass
        self.maxIters = maxIters
        self.stepSize = stepSize
        
        self.lossVals = np.zeros(self.maxIters)
        
    def run(self,initialGuess,fName,verbose=True):
        s = initialGuess.copy()
        
        t0 = time.time()
        for i in range(self.maxIters):
            self.lossVals[i] = np.real(self.lossClass.loss(s))
            realGrad, imagGrad = self.lossClass.grad(s)
            s -= self.stepSize*(realGrad + 1j*imagGrad)
            if verbose:
                print('Loss at iteration %d: %.3e'%(i,self.lossVals[i]),flush=True)
            
        t1 = time.time()
        
        self.write_results(fName,t1-t0,s)
        
        return s
    
    def write_results(self,fName,runTime,sol):
        h5File = h5py.File(fName,'a')
        
        h5File.attrs.create('method','ComplexGradientDescent')
        h5File.attrs.create('runTime',runTime)
        
        h5File.create_dataset('lossValues',data=self.lossVals)
        h5File.create_dataset('solution',data=sol)
        
        h5File.create_group('optimizerParams')
        h5File['optimizerParams'].attrs.create('stepSize',self.stepSize)
        
        h5File.close()
        return None
    
outputFile = 'test.h5'
    
sparseGridOrder = 6
if sparseGridOrder > 6:
    raise ValueError("Expect to reach desktop memory limits soon")
    
uniqueCoords, zz = read_pes('../data/UNEDF1.dat')
samplePoints, sampleEvals = SamplePoints.get_sparse_grid_2d(uniqueCoords,zz,sparseGridOrder,
                                                            outputFile,flip=True)
arrDim, sampleMatrix = SamplePoints.get_sample_matrix_2d(sparseGridOrder)

dft1d = scipy.fft.fft(np.eye(arrDim))
Psi = np.kron(dft1d,dft1d)

h5File = h5py.File(outputFile,'a')
h5File.attrs.create('basisName','dft')
h5File.close()

lamb = 500
lassoClass = ComplexLasso(lamb,sampleMatrix,Psi,sampleEvals,outputFile)

nIters = 1000
stepSize = 10**(-4)
opt = ComplexGradientDescent(lassoClass,nIters,stepSize)

s = np.random.rand(sampleMatrix.shape[1]).astype(complex)
s = opt.run(s,outputFile)
    

fig, ax = plt.subplots()
ax.plot(opt.lossVals)
ax.set(yscale='log')

fig, ax = plt.subplots()
ax.plot(np.real(s))
ax.plot(np.imag(s))

newPes = Psi @ s

rmse, newCoordVals, pesDiff = pes_rmse(uniqueCoords,zz,newPes.reshape(2*(arrDim,)),flip=True)

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

fig, ax = plt.subplots()
cf = ax.pcolormesh(np.fft.fftshift(np.imag(sReshaped)),cmap='binary')
plt.colorbar(cf,ax=ax)
