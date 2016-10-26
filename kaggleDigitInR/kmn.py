from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import csv
import numpy

class kaggleMnist(DenseDesignMatrix):
    def __init__(self , filename=None ,which_set = None , start=None, stop= None ):

        read = csv.reader(open(filename))
        #head = read.next()
        
        data = numpy.asarray([r for r in read])
        
        size = data.shape[0]
        features = data.shape[1]
        
        print data.shape

        if which_set=='train':
            y = numpy.asarray(data[start : stop , 0:1 ] , dtype=int)
            X = numpy.asarray(data[start : stop , 1:features ] , dtype=float)
            super(kaggleMnist , self).__init__(X = X,y = y , y_labels=10)
        elif which_set=='valid':
            y = numpy.asarray(data[start : stop , 0:1 ] , dtype=int)
            X = numpy.asarray(data[start : stop , 1:features ] , dtype=float)
            super(kaggleMnist , self).__init__(X = X,y = y , y_labels=10)
        else:
            y = numpy.asarray(data[:,0:1] , dtype=int)
            X = numpy.asarray(data[:,1:features] , dtype=float)
            super(kaggleMnist , self).__init__(X = X,y = y , y_labels=10)
            