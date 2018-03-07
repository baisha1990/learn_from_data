
import numpy as np
import random
import matplotlib.pyplot as plt

class PLA_14:
    def __init__(self, N, err=0):
        # Random linearly separated data
        xA,yA,xB,yB = [random.uniform(-5, 5) for i in range(4)] 
        
        #V is the weight vector with values[w0,w1,w2] generated in a random uniform manner.
        self.targetWeight = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        
        #this is the data set of N rows 
        self.dataSet = self.generate_dataset(N)
        self.classificationErr=err
 
    def generate_dataset(self, N):
        dataSet = []
        for i in range(N):
            #generate x1,x2 for ith row in data set
            feature1,feature2 = [random.uniform(-5, 5) for i in range(2)]
            
            #feature vector for ith row [x0,x1,x2]
            featureVector = np.array([1,feature1,feature2])
            
            #solution space for xi, s=sign(WT.X)
            output = int(np.sign(self.targetWeight.T.dot(featureVector)))
                       
            dataSet.append((featureVector, output))

        return dataSet
 
    def plot_graph(self, currHypoWeight=[]):
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        weightVector = self.targetWeight
        slope, intercept = -(weightVector[1]/weightVector[2]), -weightVector[0]/weightVector[2] # a=slope, b= coeff if line
        l = np.linspace(-5,5)
        plt.plot(l, slope*l+intercept, 'k--') # plot a line al+b
        cols = {1: 'g', -1: 'r'}
        for feature,output in self.dataSet:
            plt.plot(feature[1], feature[2], cols[output]+'o')     
        if len(currHypoWeight)!=0:
            hypothesis_slope  = -currHypoWeight[1]/currHypoWeight[2]
            hypothesis_intercept = -currHypoWeight[0]/currHypoWeight[2]
            plt.plot(l, hypothesis_slope*l+hypothesis_intercept, 'b-', lw=2)
          
        plt.show()
 
    def classification_error(self, currWeights):
        # Error defined as fraction of misclassified points
        datSet = self.dataSet
        M = len(datSet)
        datSet_mispts = 0
        for features,output in datSet:
            if int(np.sign(currWeights.T.dot(features))) != output:
                datSet_mispts += 1
        error = datSet_mispts / float(M)
        return error
 
    def rand_missclassfied_pt(self, currHypoWeight):
        # Choose a random point among the misclassified
        datSet = self.dataSet
        mispts = []
        for features,output in datSet:
            if int(np.sign(currHypoWeight.T.dot(features))) != output: # check if h(x) != f(x)
                mispts.append((features, output))
        return mispts[random.randrange(0,len(mispts))] # Return the missclassified point 
 
    def pla(self):
        
        currHypoWeight = np.zeros(3) # weight vector of current hypotheses
        N = len(self.dataSet)
        iterations = 0
        # Iterate until all points are correctly classified

        while self.classification_error(currHypoWeight) > self.classificationErr:
            iterations += 1
            # Pick random misclassified point
            missclass_features, missclass_output = self.rand_missclassfied_pt(currHypoWeight)
            # Update weights
            nextHypoWeight = currHypoWeight + missclass_features*missclass_output
            currHypoWeight=nextHypoWeight
            self.plot_graph(currHypoWeight)
            plt.title('N = %s, Iteration %s\n'
                          % (str(N),str(iterations)))
            
            plt.show()
                
        self.finalWeight = currHypoWeight
        self.iterations = iterations
        
    def print_dataset(self):
        for feature,output in self.dataSet:
            print(feature[1],"  ",feature[2]," :: ",output)
            
    def print_final_weights(self):   
        print("Weight vector of learned hypothesis function g is \n",self.finalWeight)
    
    def print_iterations(self):
        print("Iterations it took to converge towards the target function:   \n",self.iterations)
        
        
                

        
p = PLA_14(100,0) # N,error value -- to missclassify 3 points out of N, then pass 3/N as err value.
p.plot_graph()     #plot target function shown by black line and random points which are classified into +1(blue) and -1(red)  

p.pla() #calling perceptron model (blue line is hypothesis and black is target function)
p.print_dataset() # print dataset that is generated.
p.print_final_weights() # this is the learned weights after running PLA algorithm.
p.print_iterations()
      