import numpy as np
import os
import math
from scipy.special import comb
import subprocess
import sys


class BC:

    """
        The syntax used to yield Betti curve is class method compute_betti_curves().

        Parameters：
        ----------    
        inputMatrix: Numpy 2D array.
            The symetric matrix needed to be solved.

        maxBettiNumber: int. Larger than 0.
            The maximum Betti number excepted to be calculated. The default value is 1.

        computeBetti0: bool.
            Flag to compute 0th Betti curve. The default value is False.

        maxEdgeDensity: float. Between 0 and 1.
            The rate of computing filtration. The birth point will be nonexist if larger than the rate. The default value is 0.8.

        filePrefix: str.
            The prefix of intermediate files during calculation. The default value is "matrix".

        keepFile: bool.
            Flag to keep the intermediate files. The default value is False.

        workDirectory: path-like string.
            Path to the directory that generates the intermediate files. The default value is ".".

        baseDirectory: path-like string.
            Path to the code directory. The default value is current dir.

    """

    def __init__(self,
                inputMatrix,  # A symmetric matrix
                maxBettiNumber = 1,
                computeBetti0 = False,
                maxEdgeDensity = 0.8,
                filePrefix = "matrix",
                keepFile = False,
                workDirectory = ".",
                baseDirectory = os.path.dirname(os.path.realpath(__file__)),
                computePereusInterval = False,
                ):

        assert isinstance(inputMatrix, np.ndarray)  # check if is an array
        assert all (len (row) == len (inputMatrix) for row in inputMatrix) # check if is square
        assert np.allclose(inputMatrix, inputMatrix.T) # check if is symmetric
        assert isinstance(maxBettiNumber, int) and maxBettiNumber >= 0
        assert isinstance(computeBetti0, bool)
        assert maxEdgeDensity > 0 and maxEdgeDensity <= 1
        assert isinstance(filePrefix, str)
        assert isinstance(keepFile, bool)
        assert isinstance(workDirectory, str)   # directory of .txt files
        assert isinstance(baseDirectory, str)   # only matters pereus directory

        self.inputMatrix = inputMatrix.astype(np.float)
        self.maxBettiNumber = maxBettiNumber
        self.computeBetti0 = computeBetti0
        self.maxEdgeDensity = maxEdgeDensity
        self.filePrefix = filePrefix
        self.keepFile = keepFile
        self.workDirectory = workDirectory
        self.baseDirectory = baseDirectory
        self.perseusDirectory = os.path.join(self.baseDirectory, "perseus")
        self.computePereusInterval = computePereusInterval


    def compute_betti_curves(self):
        self.check_file()
        self.numFiltrations, self.orderCanonicalForm = self.solve_filtration()
        self.generate_txt(self.inputMatrix, self.maxBettiNumber + 2)
        self.run_perseus()
        self.bettis = self.process_results()
        if self.computePereusInterval:
            self.cal_peresistenceInv() 
        self.remove_files()

        return self.bettis


    def check_file(self):
        os.chdir(self.workDirectory)
        try:
            if os.path.isfile("%s/%s_max_simplices.txt"%(self.workDirectory, self.filePrefix)):
                raise Exception("File %s_max_simplices.txt already exists in directory %s"%(self.filePrefix, self.workDirectory)) 
            if os.path.isfile("%s/%s_simplices.txt"%(self.workDirectory, self.filePrefix)):
                raise Exception("File %s_simplices.txt already exists in directory %s"%(self.filePrefix, self.workDirectory)) 
            if os.path.isfile("%s/%s_homology_betti.txt"%(self.workDirectory, self.filePrefix)):
                raise Exception("File %s_homology_betti.txt already exists in directory %s"%(self.filePrefix, self.workDirectory)) 
            for d in range(self.maxBettiNumber + 2):
                if os.path.isfile("%s/%s_homology_%d.txt"%(self.workDirectory, self.filePrefix, d)):
                    raise Exception("File %s_homology_%d.txt already exists in directory %s"%(self.filePrefix, d, self.workDirectory)) 
                
        except Exception as e:
            print(e)
            raise e


    def solve_filtration(self):
        np.fill_diagonal(self.inputMatrix, np.inf) 
        matrixSize = np.size(self.inputMatrix, 0)
        maxFiltration = math.ceil(comb(matrixSize, 2) * self.maxEdgeDensity)
        orderCanonicalForm = np.zeros(self.inputMatrix.shape)
        orderedElements = np.unique(np.sort(self.inputMatrix, axis=None))[::-1]  # here is unique
        for i, item in enumerate(orderedElements):
            orderCanonicalForm[np.where(self.inputMatrix == item)] = i
            if i > maxFiltration:
                orderCanonicalForm[np.where(self.inputMatrix == item)] = np.inf

        return maxFiltration, orderCanonicalForm

    def generate_txt(self, symMatrix, maxCliqueSize):
        matrixSize = np.size(self.inputMatrix, 0)
        with open("%s_simplices.txt"%(self.filePrefix), "w") as f:
            f.write("1\n")
            for i in range(1, matrixSize + 1):
                f.write("0 %d 1"%(i))
                f.write("\n")

            for simplexSize in range(2, min(maxCliqueSize, matrixSize) + 1):
                idx = np.arange(0, simplexSize)   # matlab starts from 1, here 0.
                fl = np.arange(matrixSize - simplexSize, matrixSize) # matlab add 1.
                completeFlag = False

                while not completeFlag:
                    thisMinor = self.orderCanonicalForm[idx[:, np.newaxis], idx]
                    thisFilt = np.max(thisMinor)
                    # print("thisFilt: ", thisFilt)
                    if thisFilt < np.inf:
                        f.write("%d "%(simplexSize - 1))
                        [f.write("%d "%(item + 1)) for item in idx]
                        f.write("%d \n"%(thisFilt))
                        incIdx = np.where(idx < fl)[0][-1]
                    else:
                        infcol, infrow = np.where(thisMinor == np.inf)[0][-1], np.where(thisMinor == np.inf)[1][-1]
                        incIdx = min(max(infcol, infrow), np.where(idx < fl)[0][-1])

                        
                    idx[incIdx:] = np.arange(idx[incIdx] + 1, idx[incIdx] + simplexSize - incIdx + 1)

                    if idx[0] > (matrixSize - simplexSize - 1):
                        thisMinor = self.orderCanonicalForm[idx[:, np.newaxis], idx]
                        thisFilt = np.max(thisMinor)
                        if thisFilt < np.inf:
                            f.write("%d "%(simplexSize - 1))
                            [f.write("%d "%(item + 1)) for item in idx]
                            f.write("%d \n"%(thisFilt))

                        completeFlag = True


    def run_perseus(self):
        try:
            subprocess.run(["%s/perseusLin"%(self.perseusDirectory), "nmfsimtop", \
                "%s_simplices.txt"%(self.filePrefix), "%s_homology"%(self.filePrefix)], stdout=subprocess.DEVNULL)
        except Exception as e:
            print(e)
            raise Exception("Run Perseus Error!")

    def process_results(self):
        matrixSize = np.size(self.inputMatrix, 0)
        edgeDensity = np.arange(1, self.numFiltrations + 1) / comb(matrixSize, 2)
        with open("%s_homology_betti.txt"%(self.filePrefix), "r") as f:
            f.readline()
            tLine = list(map(int, f.readline().split(" ")[1 : -1]))
            numCols = len(tLine)
            bettis = np.zeros((self.numFiltrations, numCols - 1))
            bettis[tLine[0] - 1] = tLine[1:]
            for line in f.readlines():
                tempRe = list(map(int, line.split(" ")[1 : -1]))
                assert len(tempRe) > 0
                bettis[tempRe[0] - 1] = tempRe[1:]
            
        for i in range(1, self.numFiltrations):
            if bettis[i, 0] == 0:
                bettis[i, :] = bettis[i-1, :]

        if self.computeBetti0:
            bettis = bettis[:, 0:min(self.maxBettiNumber + 1, np.size(bettis, 1))]
        else:
            bettis = bettis[:, 1:min(self.maxBettiNumber + 1, np.size(bettis, 1))]

        return bettis

    
    def cal_peresistenceInv(self):

        if self.computeBetti0:
            persistenceIntervals = np.zeros((self.numFiltrations, self.maxBettiNumber + 1))
            unboundedIntervals = np.zeros((self.maxBettiNumber + 1))
            for d in range(0, self.maxBettiNumber + 1):
                persistenceIntervals[:, d], unboundedIntervals[d] = self.read_peresistence_interval("%s_homology_%d.txt"%(self.filePrefix, d))

        else:
            persistenceIntervals = np.zeros((self.numFiltrations, self.maxBettiNumber))
            unboundedIntervals = np.zeros((self.maxBettiNumber))

            for d in range(1, self.maxBettiNumber + 1):
                persistenceIntervals[:, d - 1], unboundedIntervals[d - 1] = self.read_peresistence_interval("%s_homology_%d.txt"%(self.filePrefix, d))

    def read_peresistence_interval(self, fname):
        distribution = np.zeros((self.numFiltrations))
        infinite_intervals = 0

        with open(fname, "r") as f:
            for item in f.readlines():
                interval = list(map(int, item[:-1].split(" ")))

                if interval[-1] == -1:
                    infinite_intervals = infinite_intervals + 1
                else:
                    infLen = interval[-1] - interval[0]
                    distribution[infLen - 1] = distribution[infLen - 1] + 1
         
        return distribution, infinite_intervals


    def remove_files(self):
        if not self.keepFile:
            try:
                if os.path.isfile("%s/%s_max_simplices.txt"%(self.workDirectory, self.filePrefix)):
                    os.remove("%s/%s_max_simplices.txt"%(self.workDirectory, self.filePrefix))
                if os.path.isfile("%s/%s_simplices.txt"%(self.workDirectory, self.filePrefix)):
                    os.remove("%s/%s_simplices.txt"%(self.workDirectory, self.filePrefix))
                if os.path.isfile("%s/%s_homology_betti.txt"%(self.workDirectory, self.filePrefix)):
                    os.remove("%s/%s_homology_betti.txt"%(self.workDirectory, self.filePrefix))
                for d in range(self.maxBettiNumber + 2):
                    if os.path.isfile("%s/%s_homology_%d.txt"%(self.workDirectory, self.filePrefix, d)):
                        os.remove("%s/%s_homology_%d.txt"%(self.workDirectory, self.filePrefix, d))
            except Exception as e:
                print(e)
                raise e
                    


if __name__ == "__main__":
    # a = np.random.random((3, 3))
    a = np.random.random((100, 100))
    a = a + a.T
    # a[0, 6] = 5
    # a[6, 0] = 5
    ct = Clique_Topology(a, computeBetti0=True)
    ct.compute_betti_curves()
    # print(ct.__dict__)
    