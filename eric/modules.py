import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import h5py
import pandas as pd
import sys, os
import time

from scipy.interpolate import RectBivariateSpline

#For consistency
np.random.seed(10)

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
    def __init__(self,lamb,y,fileName,sampleMatrix=None,Psi=None,CPsi=None):
        # if (sampleMatrix is None) or (Psi is None):
        #     if CPsi is None:
        #         raise ValueError
        self.lamb = lamb
        self.sampleMatrix = sampleMatrix
        self.Psi = Psi
        self.y = y

        self.CPsi = CPsi

        self.fileName = fileName

        self._setup_matrix_products()

    def _setup_matrix_products(self):
        #Can be moved elsewhere later
        if self.sampleMatrix is not None:
            C_Psi = self.sampleMatrix @ self.Psi
        else:
            C_Psi = self.CPsi
        self.PsiDagger_CT_C_Psi = np.conjugate(C_Psi).T @ C_Psi
        self.yT_C_A = self.y.T @ np.real(C_Psi)
        self.yT_C_B = self.y.T @ np.imag(C_Psi)
        self.yT_y = self.y.T @ self.y
        return None

    def loss(self,s):
        a = np.real(s)
        b = np.imag(s)

        f1 = np.linalg.norm(self.CPsi @ a)**2 + np.linalg.norm(self.CPsi@b)**2 + self.yT_y
        # f1 = a.T @ self.PsiDagger_CT_C_Psi @ a + b.T @ self.PsiDagger_CT_C_Psi @ b + self.yT_y
        f1 += -2*(self.yT_C_A @ a - self.yT_C_B @ b)

        f2 = self.lamb * np.sum(np.sqrt(a**2+b**2))
        return f1 + f2

    def _grad_f1(self,a,b):
        # grad of the | C \psi - y |_{2}^{2} term
        term1 = self.CPsi @ a
        term1 = np.conjugate(self.CPsi).T @ term1
        term1 -= self.yT_C_A

        term2 = self.CPsi @ b
        term2 = np.conjugate(self.CPsi).T @ term2
        term2 += self.yT_C_B
        return 2*term1, 2*term2
        # return 2*(self.PsiDagger_CT_C_Psi @ a - self.yT_C_A), 2*(self.PsiDagger_CT_C_Psi @ b + self.yT_C_B)

    def _grad_f2(self,a,b):
        realRet, imagRet = np.zeros(a.shape), np.zeros(a.shape)
        goodInds = np.where((a!=0)&(b!=0))
        # badInds = np.where((a==0)&(b==0))
        # print(badInds)
        # print(np.min(np.abs(a)),np.min(np.abs(b)))
        realRet[goodInds] = self.lamb*a[goodInds]/np.sqrt(a[goodInds]**2 + b[goodInds]**2)
        imagRet[goodInds] = self.lamb*b[goodInds]/np.sqrt(a[goodInds]**2 + b[goodInds]**2)
        return realRet, imagRet
        # return self.lamb*a/np.sqrt(a**2 + b**2), self.lamb*b/np.sqrt(a**2 + b**2)

    def grad(self,s):
        a = np.real(s)
        b = np.imag(s)

        f1_a, f1_b = self._grad_f1(a,b)
        f2_a, f2_b = self._grad_f2(a,b)

        return np.real(f1_a+f2_a), np.real(f1_b+f2_b)

    def hessian(self,s):
        hessOut = np.zeros((2*len(s),2*len(s)))

        hessOut[:len(s),:len(s)] = 2*self.PsiDagger_CT_C_Psi
        hessOut[len(s):,len(s):] = 2*self.PsiDagger_CT_C_Psi

        a = np.real(s)
        b = np.imag(s)

        denom = (a**2 + b**2)**(3/2)
        goodInds = np.where((a!=0)&(b!=0))

        v1 = self.lamb*b**2
        v1[goodInds] /= denom[goodInds]

        v2 = -self.lamb*a*b
        v2[goodInds] /= denom[goodInds]

        v3 = self.lamb*a**2
        v3[goodInds] /= denom[goodInds]

        hessOut[:len(s),:len(s)] += np.diag(v1)
        hessOut[:len(s),len(s):] = np.diag(v2)
        hessOut[len(s):,:len(s)] = np.diag(v2)
        hessOut[len(s):,len(s):] += np.diag(v3)

        return hessOut

    def write_statistics(self):
        h5File = h5py.File(self.fileName,'a')

        h5File.attrs.create('lossClass','ComplexLasso')
        h5File.create_group('lossParams')
        h5File['lossParams'].attrs.create('lambda',self.lamb)

        h5File.close()
        return None

def bb_step(gradF1,gradF2,s1,s2):
    deltaF = gradF1 - gradF2
    deltaS = s1 - s2

    deltaFConj = np.conjugate(deltaF)
    deltaSConj = np.conjugate(deltaS)

    denom = 2*(deltaFConj @ deltaF)
    num = deltaSConj @ deltaF + deltaFConj @ deltaS

    if denom !=0:
        return num/denom
    else:
        return None

def bb_step_2(gradF1,gradF2,s1,s2):
    deltaF = gradF1 - gradF2
    deltaS = s1 - s2

    deltaFConj = np.conjugate(deltaF)
    deltaSConj = np.conjugate(deltaS)

    num = 2*deltaSConj @ deltaS
    denom = deltaSConj @ deltaF + deltaFConj @ deltaS

    if denom != 0:
        return num/denom
    else:
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

    def run_bb(self,initialGuess,fName,verbose=True,stepSize='method_1'):
        ## runs subgradient method using barzilai-Borwein step size from HW 3
        s = initialGuess.copy()

        sMinus1 = np.zeros(s.shape,dtype=complex)
        sMinus2 = np.zeros(s.shape,dtype=complex)

        gradFMinus1 = np.zeros(s.shape,dtype=complex)
        gradFMinus2 = np.zeros(s.shape,dtype=complex)

        t0 = time.time()
        for i in range(self.maxIters):
            sMinus2 = sMinus1.copy()
            gradFMinus2 = gradFMinus1.copy()

            self.lossVals[i] = np.real(self.lossClass.loss(s))
            realGrad, imagGrad = self.lossClass.grad(s)

            sMinus1 = s.copy()
            gradFMinus1 = realGrad + 1j*imagGrad

            if stepSize == 'method_1':
                dt = bb_step(gradFMinus1,gradFMinus2,sMinus1,sMinus2)
            elif stepSize == 'method_2':
                dt = bb_step_2(gradFMinus1,gradFMinus2,sMinus1,sMinus2)
            if dt is None:
                break

            s -= dt*(realGrad + 1j*imagGrad)
            if verbose:
                print('Loss at iteration %d: %.3e'%(i,self.lossVals[i]),flush=True)

            s[np.abs(s)<10**(-16)] = 0

        t1 = time.time()

        self.write_results(fName,t1-t0,s)

        return s

    def run_line_search(self):
        raise NotImplementedError

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
class ProxGradientDescent:
    def __init__(self,lossClass,maxIters,stepSize):
        self.lossClass = lossClass
        self.maxIters = maxIters
        self.stepSize = stepSize
        self.gradf1 = lossClass._grad_f1
        self.lossVals = np.zeros(self.maxIters)
        self.lamb = lossClass.lamb

    def proxf2(self,s):
        ## prox mapping on the L1 part of the target function (f2)
        ## assumes s is complex vector
        result= np.zeros(s.shape,dtype='complex')
        # need to check abs of the complex elements of s
        for k in range(len(s)):
            if np.abs(s[k]) <= self.stepSize*self.lamb :
                result[k] = 0
            else:
                result[k] = np.abs(s[k]) - self.stepSize*self.lamb
        return result

    def run(self,initialGuess,fName,verbose=True):
        s = initialGuess.copy()
        t0 = time.time()
        for k in range(self.maxIters):
            self.lossVals[k] = np.real(self.lossClass.loss(s))
            gradTerm = self.gradf1(np.real(s),np.imag(s))[0] + 1j*self.gradf1(np.real(s),np.imag(s))[1]
            norm_s = (s - self.stepSize*gradTerm)/np.abs(s - self.stepSize*gradTerm)
            s = norm_s*self.proxf2(s - self.stepSize*gradTerm)
            if verbose:
                print('Loss at iteration %d: %.3e'%(k,self.lossVals[k]),flush=True)
        t1 = time.time()

        self.write_results(fName,t1-t0,s)

        return s

    def write_results(self,fName,runTime,sol):
        h5File = h5py.File(fName,'a')

        h5File.attrs.create('method','ProximalGradientDescent')
        h5File.attrs.create('runTime',runTime)

        h5File.create_dataset('lossValues',data=self.lossVals)
        h5File.create_dataset('solution',data=sol)

        h5File.create_group('optimizerParams')
        h5File['optimizerParams'].attrs.create('stepSize',self.stepSize)

        h5File.close()

class AccelerateProxGradientDescent:
    def __init__(self,lossClass,maxIters,stepSize):
        self.lossClass = lossClass
        self.maxIters = maxIters
        self.stepSize = stepSize
        self.gradf1 = lossClass._grad_f1
        self.lossVals = np.zeros(self.maxIters)
        self.lamb = lossClass.lamb

    def proxf2(self,s):
        ## prox mapping on the L1 part of the target function (f2)
        result= np.zeros(s.shape)
        for k in range(len(s)):
            if abs(s[k]) <= self.stepSize*self.lamb :
                result[k] = 0
            else:
                result[k] = s[k] - self.stepSize*np.sign(s[k])
        return result

    def run(self,initialGuess,fName,verbose=True):
        s = initialGuess.copy()
        t0 = time.time()
        for k in range(self.maxIters):
            s = self.proxf2(s - self.stepSize*self.gradf1(np.real(s),np.imag(s)))
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

class ComplexNewtonsMethod:
    def __init__(self,lossClass,maxIters,stepSize):
        self.lossClass = lossClass
        self.maxIters = maxIters
        self.stepSize = stepSize

        self.lossVals = np.zeros(self.maxIters)

def make_CPsi(sampleMatrix,dft1d):
    n = dft1d.shape[0]
    ret = np.zeros((sampleMatrix.shape[0],n**2),dtype=complex)

    rng = np.arange(n**2,dtype=int)

    for i in range(ret.shape[0]):
        idx = np.where(sampleMatrix[i]!=0)[0]
        ret[i] = dft1d[idx//n,rng//n] * dft1d[idx%n,rng%n]
        # for k in range(n**2):
        #     # print(i,idx,k)
        #     ret[i,k] = dft1d[idx//n,k//n] * dft1d[idx%n,k%n]
    return ret
