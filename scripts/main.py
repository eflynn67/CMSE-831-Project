import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
# import h5py
import pandas as pd
import sys, os

from scipy.interpolate import RectBivariateSpline

#Maybe measurements should be random, rather than structured...?

def get_sample_points(sparse2d,fName='UNEDF1.dat'):
    pes = pd.read_csv('../data/'+fName,sep='\t')
    pes['EHFB'] -= pes['EHFB'].mean()
    uniqueCoords = [np.unique(pes[col]) for col in ['Q20','Q30']]
    zz = pes['EHFB'].to_numpy().reshape([len(u) for u in uniqueCoords])
    
    
    zz = np.vstack((np.flip(zz[1:],axis=0),zz))
    zz = np.hstack((np.flip(zz[:,1:],axis=1),zz))
    
    
    xFlipped = np.concatenate((-np.flip(uniqueCoords[0][1:]),uniqueCoords[0]))
    yFlipped = np.concatenate((-np.flip(uniqueCoords[1][1:]),uniqueCoords[1]))
    
    fig, ax = plt.subplots()
    ax.contourf(xFlipped,yFlipped,zz.T.clip(-30,30),cmap='Spectral_r',levels=30)
    
    pes_interp = RectBivariateSpline(xFlipped,yFlipped,zz)
    
    sparse2d[:,0] = sparse2d[:,0]*(xFlipped[-1] - xFlipped[0]) + xFlipped[0]
    sparse2d[:,1] = sparse2d[:,1]*(yFlipped[-1] - yFlipped[0]) + yFlipped[0]
    
    ax.scatter(*sparse2d.T,color='black')
    
    fig.savefig('../plots/pes-flipped.pdf',bbox_inches='tight')
    
    return pes_interp(*sparse2d.T,grid=False)

def subset_sum(numbers, target, partial=[], partial_sum=0):
    #From https://stackoverflow.com/a/4633515
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)

def sparse_1d(n):
    arr = np.arange(0,2**n)
    return 2**(-n)*arr

def sparse_2d(n):
    sparseArrList = []
    
    listOf1dVals = [sparse_1d(i) for i in range(n+1)]
    
    listOfCombinations = [l for l in list(subset_sum(range(n+1),n)) if len(l)==2]
    
    for l in listOfCombinations:
        sparseArrList.append(list(itertools.product(listOf1dVals[l[0]],listOf1dVals[l[1]])))
        sparseArrList.append(list(itertools.product(listOf1dVals[l[1]],listOf1dVals[l[0]])))
    
    arrOut = np.array(sparseArrList).reshape((-1,2))
    
    return np.unique(arrOut,axis=0)

class ComplexLasso:
    def __init__(self,lamb,sampleMatrix,Psi,y):
        self.lamb = lamb
        self.sampleMatrix = sampleMatrix
        self.Psi = Psi
        self.y = y
        
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
    
sparseGridOrder = 6
if sparseGridOrder > 6:
    raise ValueError("Expect to reach desktop memory limits soon")
sparse2d = sparse_2d(sparseGridOrder)
fullArrSize1d = len(sparse2d[sparse2d[:,0]==0]) #Number of elements along one axis

dft1d = scipy.fft.fft(np.eye(fullArrSize1d))
Psi = np.kron(dft1d,dft1d)
A = np.real(Psi)
B = np.imag(Psi)

sampleMatrix = np.zeros((len(sparse2d),fullArrSize1d,fullArrSize1d))
sparseGridIndices = (sparse2d * 2**sparseGridOrder).astype(int)
for (rowIter,row) in enumerate(sparseGridIndices):
    sampleMatrix[(rowIter,)+tuple(row)] = 1
sampleMatrix = sampleMatrix.reshape((len(sparse2d),-1))

samplePoints = get_sample_points(sparse2d)
# sys.exit()
# samplePoints = samplePoints[::-30]
# sampleMatrix = sampleMatrix[::-30]

lassoClass = ComplexLasso(500,sampleMatrix,Psi,samplePoints)

nIters = 5000
lossArr = np.zeros(nIters)
stepSize = 10**(-4)
s = np.random.rand(sampleMatrix.shape[1]).astype(complex)

for i in range(nIters):
    lossArr[i] = np.real(lassoClass.loss(s))
    print(i,'%.3e'%lossArr[i],flush=True)
    if i%250 == 0:
        newPes = Psi @ s
        fig, ax = plt.subplots()
        cf = ax.contourf(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).T.clip(-30,30),
                          cmap='Spectral_r',levels=30)
        # cf = ax.pcolormesh(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).clip(-30,30),
        #                    cmap='Spectral_r')
        plt.colorbar(cf,ax=ax)
        fig.savefig('../plots/pes-flipped-iteration-'+str(i).zfill(4)+'.pdf',bbox_inches='tight')
    realGrad, imagGrad = lassoClass.grad(s)
    s -= stepSize * (realGrad + 1j*imagGrad)
    
#%%
fig, ax = plt.subplots()
ax.plot(lossArr)
ax.set(yscale='log')
#%%
fig, ax = plt.subplots()
ax.plot(np.real(s))
ax.plot(np.imag(s))

#%%
newPes = Psi @ s
fig, ax = plt.subplots()
cf = ax.contourf(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).T.clip(-30,30),
                  cmap='Spectral_r',levels=30)
# cf = ax.pcolormesh(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).clip(-30,30),
#                    cmap='Spectral_r')
plt.colorbar(cf,ax=ax)
fig.savefig('../plots/pes-flipped-iteration-'+str(nIters).zfill(4)+'.pdf',bbox_inches='tight')

fig, ax = plt.subplots()
cf = ax.contourf(np.imag(newPes).reshape((fullArrSize1d,fullArrSize1d)).T,
                 cmap='Spectral_r',levels=30)
plt.colorbar(cf,ax=ax)

#%%
sReshaped = s.reshape(2*(fullArrSize1d,))
fig, ax = plt.subplots()
cf = ax.pcolormesh(np.fft.fftshift(np.real(sReshaped)),cmap='binary')
plt.colorbar(cf,ax=ax)

fig, ax = plt.subplots()
cf = ax.pcolormesh(np.fft.fftshift(np.imag(sReshaped)),cmap='binary')
plt.colorbar(cf,ax=ax)
# #%%
# fig, ax = plt.subplots()
# ax.scatter(*sparse2d.T,c=samplePoints,cmap='Spectral_r')

# #%%
# fig, ax = plt.subplots()
# ax.pcolormesh(np.real(Psi),cmap='binary')
# ax.pcolormesh(np.imag(Psi),cmap='binary')
