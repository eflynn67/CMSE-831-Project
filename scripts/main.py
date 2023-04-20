import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import h5py
import pandas as pd
import sys, os
import time
import scipy.fftpack
import argparse

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
        # print(sparseGridIndices)
        for (rowIter,row) in enumerate(sparseGridIndices):
            sampleMatrix[(rowIter,)+tuple(row)] = 1
        sampleMatrix = sampleMatrix.reshape((len(sparse2d),-1))

        return fullArrSize1d, sampleMatrix

    @staticmethod
    def get_suggested_initial_guess(sparseGridOrder):
        sparse2d = SamplePoints.sparse_2d(sparseGridOrder)

        fullArrSize1d = len(sparse2d[sparse2d[:,0]==0]) #Number of elements along one axis

        coordinateArr = np.zeros(2*(fullArrSize1d,))

        sparseGridIndices = (sparse2d * 2**sparseGridOrder).astype(int)

        coordinateArr[tuple(sparseGridIndices.T)] = 1

        frequencyArr = scipy.fft.fft2(coordinateArr)
        frequencyArr[np.where(frequencyArr!=0)] = 1 + 0*1j

        return frequencyArr.flatten()

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
        raise NotImplementedError('Expect to run into memory issues b/c of large Hessian')
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

def approximate_line_search(CPsi,yT_C_A,yT_C_B,a,b,dFda,dFdb,lamb):
    CPsi_conj_trans = np.conjugate(CPsi).T

    denom = a**2 + b**2
    denom[denom==0] = 10**(-4)
    #Unimportant - will be zeroed out b/c we always have a/denom or b/denom.
    #Just prevents runtime errors

    term1 = CPsi@a
    term1 = CPsi_conj_trans @ term1
    va = 2*term1 - 2*yT_C_A + lamb*a/np.sqrt(denom)

    term2 = CPsi@b
    term2 = CPsi_conj_trans @ term2
    vb = 2*term2 + 2*yT_C_B + lamb*a/np.sqrt(denom)

    term3 = CPsi@dFda
    wa = -2*CPsi_conj_trans@term3 + lamb*b*(b*dFda - a*dFdb)/denom**(3/2)

    term4 = CPsi@dFdb
    wb = -2*CPsi_conj_trans@term4 + lamb*a*(a*dFdb - b*dFda)/denom**(3/2)

    va = va.astype(float)
    vb = vb.astype(float)
    wa = wa.astype(float)
    wb = wb.astype(float)

    const = a@va + b@vb
    linearTerm = a@wa + b@wb - va@dFda - vb@dFdb

    if linearTerm != 0:
        return -const/linearTerm

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

        self.write_results(fName,t1-t0,s,'constant',None)

        return s

    def run_bb(self,initialGuess,fName,verbose=True,stepSize='method_1'):
        s = initialGuess.copy()

        sMinus1 = np.zeros(s.shape,dtype=complex)
        sMinus2 = np.zeros(s.shape,dtype=complex)

        gradFMinus1 = np.zeros(s.shape,dtype=complex)
        gradFMinus2 = np.zeros(s.shape,dtype=complex)

        dtArr = np.zeros(self.maxIters)

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
                print('Breaking: dt is None')
                break

            # dt = max(dt,dtMin)
            dtArr[i] = dt

            s -= dt*(realGrad + 1j*imagGrad)
            if verbose:
                print('Loss at iteration %d: %.3e'%(i,self.lossVals[i]),flush=True)

            s[np.abs(s)<10**(-16)] = 0

        t1 = time.time()

        self.write_results(fName,t1-t0,s,'bb_'+stepSize,dtArr)

        return s, dtArr

    def run_approximate_line_search(self,initialGuess,fName,verbose=True):
        s = initialGuess.copy()

        dtArr = np.zeros(self.maxIters)

        t0 = time.time()
        for i in range(self.maxIters):
            # print(50*'-')
            self.lossVals[i] = np.real(self.lossClass.loss(s))
            realGrad, imagGrad = self.lossClass.grad(s)

            dt = approximate_line_search(self.lossClass.CPsi,self.lossClass.yT_C_A,
                                         self.lossClass.yT_C_B,np.real(s),np.imag(s),
                                         realGrad,imagGrad,self.lossClass.lamb)

            if dt is None:
                print('Breaking: dt is None')
                break

            dtArr[i] = dt

            s -= dt*(realGrad + 1j*imagGrad)
            if verbose:
                print('Loss, dt at iteration %d: %.3e %.3e'%(i,self.lossVals[i],dt),flush=True)

        t1 = time.time()

        self.write_results(fName,t1-t0,s,'approx_line_search',dtArr)

        return s, dtArr

    def write_results(self,fName,runTime,sol,dtMethod,dtArr):
        h5File = h5py.File(fName,'a')

        h5File.attrs.create('method','ComplexGradientDescent')
        h5File.attrs.create('runTime',runTime)

        h5File.create_dataset('lossValues',data=self.lossVals)
        h5File.create_dataset('solution',data=sol)

        h5File.create_group('optimizerParams')
        if dtMethod == 'constant':
            h5File['optimizerParams'].attrs.create('stepSize',self.stepSize)
        else:
            h5File['optimizerParams'].create_dataset('dtArr',data=dtArr)
        h5File['optimizerParams'].attrs.create('dtMethod',dtMethod)

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
class AccProxGradientDescent:
    def __init__(self,lossClass,maxIters,stepSize,m,gamma0):
        self.lossClass = lossClass
        self.maxIters = maxIters
        self.stepSize = stepSize
        self.gradf1 = lossClass._grad_f1
        self.lossVals = np.zeros(self.maxIters)
        self.lamb = lossClass.lamb
        self.gamma0 = gamma0
        self.m = m
    def _calcTheta(self,gamma):
        return 0.5*(self.m - gamma)*self.stepSize + np.sqrt(gamma*self.stepSize + 0.25*(self.m-gamma)**2 * self.stepSize**2)
    def _calcGamma(self,theta):
        return (theta**2)/self.stepSize
    def _calcV(self,s1,v1,theta):
        v2 = s1 + ((theta- self.m*self.stepSize)/(1-self.m*self.stepSize))*(v1 -s1)
        return v2
    def _calcY(self,theta,v1,s1):
        y = s1 + ((theta- self.m*self.stepSize)/(1-self.m*self.stepSize))*(v1 -s1)
        return y
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
        v1 = np.zeros(s.shape,dtype='complex')
        gamma = self.gamma0
        for k in range(self.maxIters):
            self.lossVals[k] = np.real(self.lossClass.loss(s))
            theta = self._calcTheta(gamma)

            y = self._calcY(theta,v1,s)
            gradTerm = self.gradf1(np.real(y),np.imag(y))[0] + 1j*self.gradf1(np.real(y),np.imag(y))[1]
            norm_s = (s - self.stepSize*gradTerm)/np.abs(s - self.stepSize*gradTerm)
            s = norm_s*self.proxf2(s - self.stepSize*gradTerm)

            v1 = self._calcV(s,v1,theta)
            gamma = self._calcGamma(theta)
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


def main(lamb, stepSize, sparseGridOrder, maxIterations):
      
  parentPath = os.path.dirname(os.getcwd())
  savePath = parentPath + '/data/output' 
  outputFile = (savePath + "/output_lambda" + str(lamb) + "_stepSize" 
                + str(stepSize) + "_sparseGridOrder" + str(sparseGridOrder) 
                + "_maxIterations" + str(maxIterations) + ".h5")
  
  try:
      os.remove(outputFile)
  except FileNotFoundError:
      pass
  
  if sparseGridOrder > 6:
      raise ValueError("Expect to reach desktop memory limits soon")
  
  uniqueCoords, zz = read_pes('../data/UNEDF1.dat')
  samplePoints, sampleEvals = SamplePoints.get_sparse_grid_2d(uniqueCoords,zz,sparseGridOrder,
                                                              outputFile,flip=True)
  
  arrDim, sampleMatrix = SamplePoints.get_sample_matrix_2d(sparseGridOrder)
  
  dft1d = scipy.fft.fft(np.eye(arrDim))
  # dft1d = np.fft.fftshift(scipy.fft.fft(np.eye(arrDim)))
  CPsi = make_CPsi(sampleMatrix,dft1d)
  Psi = np.kron(dft1d,dft1d)
  
  # print(sampleMatrix @ Psi - CPsi)
  # sys.exit()
  
  h5File = h5py.File(outputFile,'a')
  h5File.attrs.create('basisName','dft')
  h5File.close()
  
  # lamb = 500
  # lassoClass = ComplexLasso(lamb,sampleEvals,outputFile,sampleMatrix,Psi)
  lassoClass = ComplexLasso(lamb,sampleEvals,outputFile,CPsi=CPsi)
  
  opt = ComplexGradientDescent(lassoClass,maxIterations,stepSize)
  
  s = SamplePoints.get_suggested_initial_guess(sparseGridOrder)
  # s = np.random.rand(sampleMatrix.shape[1]).astype(complex)
  # s = np.zeros(sampleMatrix.shape[1]).astype(complex)
  
  rt0 = time.time()
  s = opt.run(s,outputFile)
  # s, dtArr = opt.run_bb(s,outputFile,stepSize='method_2')
  # s, dtArr = opt.run_approximate_line_search(s,outputFile)
  rt1 = time.time()
  print("Run time: %.6f"%(rt1-rt0))
  
  fig, ax = plt.subplots()
  ax.plot(opt.lossVals)
  ax.set(yscale='log')
  
  fig, ax = plt.subplots()
  ax.plot(np.real(s))
  ax.plot(np.imag(s))
  
  # fig, ax = plt.subplots()
  # ax.plot(dtArr)
  # ax.set(yscale='log',title='dt vs iteration')
  
  #%%
  
  t0 = time.time()
  newPes = Psi @ s
  t1 = time.time()
  print('Matrix multiplication time: %.6f'%(t1-t0))
  # newPes = np.fft.fftshift(np.fft.fft(s))
  
  t0 = time.time()
  newPes = scipy.fft.fft2(s.reshape(2*(arrDim,)))
  t1 = time.time()
  print('FFT time: %.6f'%(t1-t0))
  
  rmse, newCoordVals, pesDiff = pes_rmse(uniqueCoords,zz,newPes.reshape(2*(arrDim,)),flip=True)
  
  h5File = h5py.File(outputFile,'a')
  h5File.attrs.create('pesRMSE',rmse)
  h5File.close()
  #%%
  # fig, ax = plt.subplots()
  # cf = ax.contourf(*newCoordVals,pesDiff.T.clip(-5,5),cmap='Spectral_r',levels=30)
  # plt.colorbar(cf,ax=ax)
  # ax.set(title='PES Difference')
  
  fig, ax = plt.subplots()
  cf = ax.contourf(np.real(newPes).reshape((arrDim,arrDim)).T.clip(-30,30),
                    cmap='Spectral_r',levels=30)
  # cf = ax.pcolormesh(np.real(newPes).reshape((fullArrSize1d,fullArrSize1d)).clip(-30,30),
  #                    cmap='Spectral_r')
  plt.colorbar(cf,ax=ax)
  ax.set(title='Fit PES Real Component')
  
  # fig, ax = plt.subplots()
  # cf = ax.contourf(np.imag(newPes).reshape((arrDim,arrDim)).T.clip(-1,1),
  #                  cmap='Spectral_r',levels=30)
  # plt.colorbar(cf,ax=ax)
  # ax.set(title='Fit PES Imaginary Component')
  
  # #%%
  # sReshaped = s.reshape(2*(arrDim,))
  # fig, ax = plt.subplots()
  # cf = ax.pcolormesh(np.fft.fftshift(np.real(sReshaped)),cmap='binary')
  # plt.colorbar(cf,ax=ax)
  
  # fig, ax = plt.subplots()
  # cf = ax.pcolormesh(np.fft.fftshift(np.imag(sReshaped)),cmap='binary')
  # plt.colorbar(cf,ax=ax)
  
  return

def getcommmandlineinputs(args):

  defaultInputs  = {
    "lamb" : 1000,  
    "stepSize" : 10**(-4), 
    "sparseGridOrder" : 5,
    "maxIterations" : 5000,
    } 
  
  parser = argparse.ArgumentParser(description=("Input arguments: --lamb =... "
                                     +"--stepSize=... --sparseGridOrder=... --maxIterations=..."))
  
  parser.add_argument("--lamb", required = False, type = float, default = defaultInputs["lamb"],
                         help = "Larger value enforces sparsity of Fourier transformed PES.")
  
  parser.add_argument("--stepSize", required = False, type = float, default = defaultInputs["stepSize"],
                         help = "Lower value for less accumulated error in solver.")
   
  parser.add_argument("--sparseGridOrder", required = False, type = int, default = defaultInputs["sparseGridOrder"],
                         help = "Determines mesh size of PES.")
  
  parser.add_argument("--maxIterations", required = False, type = int, default = defaultInputs["maxIterations"],
                         help = "Max iterations that solver can run.")
  
  args = parser.parse_args(args)
  defaultInputs["lamb"] = args.lamb
  defaultInputs["stepSize"] = args.stepSize
  defaultInputs["sparseGridOrder"] = args.sparseGridOrder
  defaultInputs["maxIterations"] = args.maxIterations
  
  return defaultInputs
  

if __name__ == "__main__":
   
  inputs = getcommmandlineinputs(sys.argv[1:])
  main(inputs["lamb"], inputs["stepSize"], inputs["sparseGridOrder"], inputs["maxIterations"])
