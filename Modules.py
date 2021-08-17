"""
Modules.py

Purpose: Defines the classes needed for the neural network: graph, node, edges, ...

Version: Continue investigating RMSProp step size
"""

from collections import OrderedDict
import numpy as np
import math
import logging
import pathos
import scipy.special as scp

def tensorMult(A, B):
    """Multiplies mxnxk tensor A and kxi tensor B yielding mxnxi tensor.
    Or multiplies mxn matrix A and ixnxj tensor B yielding ixmxj tensor. If
    after multiplication second dimension is 1 it is squeezed."""
    if np.ndim(A)==3:
        res=np.zeros((A.shape[0],A.shape[1],B.shape[1]))
        for i,matrix in enumerate(A):
            res[i]=np.dot(A[i],B)
        if res.shape[1]==1:
            res=res.squeeze(1)
        return res
    elif np.ndim(B)==3:
        res=np.zeros((B.shape[0],A.shape[0],B.shape[2]))
        for i,matrix in enumerate(B):
            res[i]=np.dot(A,B[i])
        if res.shape[1]==1:
            res=res.squeeze(1)
        return res
    elif np.shape(A) == (1,1) or np.shape(B) == (1,1):
        return A*B
    return np.dot(A,B)
        
class Node:
    def __init__(self):
        self.inputs=OrderedDict() #value/gradient
        self.outputs=OrderedDict() #value
        self.parameters=OrderedDict() #value/gradient
        self.additionalInputs=OrderedDict() #value
    
    def has(self, variableName):
        return variableName in self.parameters or variableName in self.inputs
    
    def setAdditionalInputs(self,namedInputs):
        for name,value in namedInputs:
            self.additionalInputs[name]=value
    
    def returnGradient(self, variableName):
        if not self.has(variableName):
            raise Exception('Node does not contain variable.')
        if variableName in self.parameters:
            return self.parameters[variableName].gradient
        else:
            return self.inputs[variableName].gradient
    
class ReLUNode(Node):
    """Defines the class of ReLu nodes."""
    def __init__(self, weight, bias):
        super().__init__()
        self.parameters['weight']=ValueGradient(weight)
        self.parameters['bias']=ValueGradient(bias)
        self.inputs['input']=ValueGradient()
        self.wl,self.wlmin1=weight.shape
        self.outputs['output']=None
    
    def ReLU(self, x):
        """ReLU function after affine transformation with own paramters."""
        return np.maximum(0,self.parameters['weight'].value@x+self.parameters['bias'].value)
        
    def forward(self, inputs):
        """Performs forward propagation and stores obtained values."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if inputs.shape != (self.wlmin1,1):
            raise Exception('Inputs do not have the correct dimensions. '
                            'Perhaps transform to column vector.') 
        self.inputs['input'].value=inputs
        self.outputs['output']=self.ReLU(inputs)
        
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        s=self.outputs['output']>0
        self.parameters['bias'].gradient=np.eye(self.wl)*s
        self.inputs['input'].gradient=self.parameters['weight'].value*s
        D=np.zeros((self.wlmin1,self.wl,self.wl))
        for i,j in enumerate(self.inputs['input'].value):
            D[i,:,:]=np.diag(np.resize(j,self.wl))*s
        self.parameters['weight'].gradient=D

class ExpNode(Node):
    """Defines the class of exponent nodes. Only exponentiate the indices
    indicated."""
    def __init__(self, indices):
        super().__init__()
        self.inputs['input']=ValueGradient()
        self.outputs['output']=None
        self.indices=indices
        
    def forward(self, inputs):
        """Performs forward propagation and stores obtained values."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        self.inputs['input'].value=inputs
        res=inputs
        res[self.indices]=np.exp(res[self.indices])
        self.outputs['output']=res
        
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        res=np.eye(len(self.inputs['input'].value))
        for i in self.indices:
            res[i,i]=np.exp(self.inputs['input'].value[i])
        self.inputs['input'].gradient=res       

class AffineNode(Node):
    """Defines the class of affine nodes without non-linearity."""
    def __init__(self, weight, bias):
        super().__init__()
        self.parameters['weight']=ValueGradient(weight)
        self.parameters['bias']=ValueGradient(bias)
        self.inputs['input']=ValueGradient()
        self.wl,self.wlmin1=weight.shape
        self.outputs['output']=None
        
    def forward(self, inputs):
        """Performs forward propagation and stores obtained values."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if inputs.shape != (self.wlmin1,1):
            raise Exception('Inputs do not have the correct dimensions. '
                            'Perhaps transform to column vector.') 
        self.inputs['input'].value=inputs
        self.outputs['output']=self.parameters['weight'].value@inputs+self.parameters['bias'].value
        
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        self.parameters['bias'].gradient=np.eye(self.wl)
        self.inputs['input'].gradient=self.parameters['weight'].value
        D=np.zeros((self.wlmin1,self.wl,self.wl))
        for i,j in enumerate(self.inputs['input'].value):
            D[i,:,:]=np.diag(np.resize(j,self.wl))
        self.parameters['weight'].gradient=D
        
class AverageNegativeLogNormalNode(Node):
    """Defines the class normal density nodes from the factor model. Factors
    is Tx1 matrix of factor. R is Tx1 vetor of to be explained returns."""
    def __init__(self):
        super().__init__()
        self.outputs['output']=None
        self.inputs['input']=ValueGradient()
        self.additionalInputs['factor']=None
        self.additionalInputs['R']=None
    
    def forward(self, inputs):
        """Performs forward propagation. Factors is Tx1 matrix of factor.
        R is Tx1 vetor of to be explained returns."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if self.additionalInputs['factor'] is None or self.additionalInputs['R'] is None:
            raise Exception('Initialise factor and return before calling forward.')
        self.inputs['input'].value=inputs
        alpha=inputs[0]
        beta=inputs[1]
        sigma=inputs[2]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        self.outputs['output']=0.5*np.log(2*math.pi)+np.log(abs(sigma))+np.mean((
            R-alpha-factor*beta)**2/(2*sigma**2))
    
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        alpha=(self.inputs['input'].value)[0]
        beta=(self.inputs['input'].value)[1]
        sigma=(self.inputs['input'].value)[2]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        da=np.mean(-(R-alpha-factor*beta)/(sigma**2))
        db=np.mean(-factor*(R-alpha-factor*beta)/(sigma**2))
        ds=np.mean(1/abs(sigma)-(R-alpha-factor*beta)**2/(sigma**3))
        self.inputs['input'].gradient=np.array([[da,db,ds]])
        
class AvNegLogNormalNodeLogSigma(Node):
    """Defines the class normal density nodes from the factor model. Factors
    is Tx1 matrix of factor. R is Tx1 vetor of to be explained returns.
    Takes log sigma as input."""
    def __init__(self):
        super().__init__()
        self.outputs['output']=None
        self.inputs['input']=ValueGradient()
        self.additionalInputs['factor']=None
        self.additionalInputs['R']=None
    
    def forward(self, inputs):
        """Performs forward propagation. Factors is Tx1 matrix of factor.
        R is Tx1 vetor of to be explained returns."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if self.additionalInputs['factor'] is None or self.additionalInputs['R'] is None:
            raise Exception('Initialise factor and return before calling forward.')
        self.inputs['input'].value=inputs
        alpha=inputs[0]
        beta=inputs[1]
        sstar=inputs[2]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        self.outputs['output']=0.5*np.log(2*math.pi)+sstar+np.mean((
            R-alpha-factor*beta)**2/(2*np.exp(2*sstar)))
    
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        alpha=(self.inputs['input'].value)[0]
        beta=(self.inputs['input'].value)[1]
        sstar=(self.inputs['input'].value)[2]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        da=np.mean(-(R-alpha-factor*beta)/(np.exp(2*sstar)))
        db=np.mean(-factor*(R-alpha-factor*beta)/(np.exp(2*sstar)))
        ds=1+np.mean(-(R-alpha-factor*beta)**2/(np.exp(2*sstar)))
        self.inputs['input'].gradient=np.array([[da,db,ds]])
        
class AvNegLogtNodeLogSigmaLogNu(Node):
    """Defines the class t density nodes from the factor model. Factors
    is Tx1 matrix of factor. R is Tx1 vetor of to be explained returns.
    Takes log sigma and log nu as input."""
    def __init__(self):
        super().__init__()
        self.outputs['output']=None
        self.inputs['input']=ValueGradient()
        self.additionalInputs['factor']=None
        self.additionalInputs['R']=None
    
    def forward(self, inputs):
        """Performs forward propagation. Factors is Tx1 matrix of factor.
        R is Tx1 vetor of to be explained returns."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if self.additionalInputs['factor'] is None or self.additionalInputs['R'] is None:
            raise Exception('Initialise factor and return before calling forward.')
        self.inputs['input'].value=inputs
        alpha=inputs[0]
        beta=inputs[1]
        s=inputs[2]
        n=inputs[3]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        self.outputs['output']=(s-np.log(scp.gamma(0.5*(np.exp(n)+1)))+
                                0.5*n +0.5*np.log(np.pi)+np.log(scp.gamma(
                                0.5*np.exp(n))+0.5*(np.exp(n)+1)*np.log(1+
                                np.mean((R-alpha-beta*factor)**2)/np.exp(n +2*s))))
                                                                        
    
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        alpha=(self.inputs['input'].value)[0]
        beta=(self.inputs['input'].value)[1]
        s=(self.inputs['input'].value)[2]
        n=(self.inputs['input'].value)[3]
        factor=self.additionalInputs['factor']
        R=self.additionalInputs['R']
        da=np.mean(-((np.exp(n)+1)*(R-alpha-beta*factor))/(np.exp(n+2*s)+(R-alpha-beta*factor)**2))
        db=np.mean(-factor*((np.exp(n)+1)*(R-alpha-beta*factor))/(np.exp(n+2*s)+(R-alpha-beta*factor)**2))
        ds=1-np.mean(((np.exp(n)+1)*(R-alpha-beta*factor)**2)/(np.exp(n+2*s)+(R-alpha-beta*factor)**2))
        dn=np.mean(-scp.digamma(0.5*(np.exp(n)+1))*0.5*np.exp(n)+0.5 +0.5*np.exp(n)*scp.digamma(
            0.5*np.exp(n))+0.5*np.exp(n)*np.log(1+(R-alpha-beta*factor)**2/np.exp(n +2*s))-0.5*(np.exp(n)+1)*(
            R-alpha-beta*factor)**2/(np.exp(n+s*2)+(R-alpha-beta*factor)**2))
        self.inputs['input'].gradient=np.array([[da,db,ds,dn]])
        
class quadraticLossNode(Node):
    def __init__(self):
        super().__init__()
        self.outputs['output']=None
        self.inputs['input']=ValueGradient()
        self.additionalInputs['y']=None
        
    def forward(self, inputs):
        """Performs forward propagation. Factors is Tx1 matrix of factor.
        R is Tx1 vetor of to be explained returns."""
        if isinstance(inputs, list):
            inputs=inputs[0]
        if self.additionalInputs['y'] is None:
            raise Exception('Initialise y before calling forward.')
        self.inputs['input'].value=inputs
        self.outputs['output']=(self.additionalInputs['y']-inputs)**2
    
    def backward(self):
        """Performs back propagation and stores the obtained gradients."""
        if self.outputs['output'] is None:
            raise Exception('Forward must be called before backward.')
        y=self.additionalInputs['y']
        yHat=self.inputs['input'].value
        self.inputs['input'].gradient=(-2*(y-yHat)).reshape((1,1))

class ValueGradient:
    def __init__(self, value=None, gradient=None):
        self.value=value
        self.gradient=gradient

class Edges:
    """"Represents edges of directed graph. edgeList is list of tuples, where
    direction runs from first index to second. Edges are represented in a
    matrix where direction runs from row to column. Nodes are numberd from
    zero."""
    def __init__(self, edgeList):
        if edgeList:
            edgeList=edgeList if isinstance(edgeList, list) else [edgeList]
            self.numberOfNodes=np.amax(edgeList)+1
            self.edges=np.zeros((self.numberOfNodes,self.numberOfNodes))
            for i in edgeList:
                self.edges[i]=1
            self.numberOfEdges=np.count_nonzero(self.edges)
        else:
            self.numberOfNodes=0
            self.numberOfEdges=0
            self.edges=np.array([])
    
    def parents(self,i):
        """Gives the parents of node i."""
        return np.nonzero(self.edges[:,i]==1)[0]
        
    def children(self,i):
        """Gives the children of node i."""
        return np.nonzero(self.edges[i,:]==1)[0]
    
    def add(self,newEdges):
        """Add the new edges to edges."""
        newEdges=newEdges if isinstance(newEdges, list) else [newEdges]
        newMax=np.amax(newEdges)+1
        if newMax>self.numberOfNodes:
            oldEdges=self.edges
            self.edges=np.zeros((newMax,newMax))
            self.edges[:self.numberOfNodes,:self.numberOfNodes]=oldEdges
            self.numberOfNodes=newMax
        for i in newEdges:
            self.edges[i]=1
        self.numberOfEdges=np.count_nonzero(self.edges)
    
    def remove(self,removeEdges):
        """Removes the edge from edges and slims down matrix of edges if allowed."""
        removeEdges=removeEdges if isinstance(removeEdges, list) else [removeEdges]
        if np.amax(removeEdges)>self.numberOfNodes:
            raise Exception("Node not in network.")
        for i in removeEdges:
            self.edges[i]=0        
        while any(self.edges[:,-1])==0 and any(self.edges[-1,:])==0:
            self.edges=self.edges[:-1,:-1]
        self.numberOfNodes=self.edges.shape[0]
        self.numberOfEdges=np.count_nonzero(self.edges)
        
    def DAGFurtherParents(self,i):
        """Gives higher generational parents of node i in a directed acyclical graph"""
        parents=self.parents(i)
        for parent in parents:
            parents=np.append(parents,self.DAGFurtherParents(parent))
        return np.unique(parents)
    
    def DAGFurtherChildren(self,i):
        """Gives higher generational children of node i in a directed acyclical graph"""
        children=self.children(i)
        for child in children:
            children=np.append(children,self.DAGFurtherChildren(child))
        return np.unique(children)
        
class Graph:
    def __init__(self, rng=None):
        self.nodes=[]
        self.edges=Edges([])
        if rng is None:
            self.rng=np.random.default_rng(28051998)
        else:
            self.rng=rng
        
    def forward(self,listedInputs,listedNamedAdditionalInputs=None):
        """Sets additional inputs and feeds inputs and through the network
        as indicated by listedInputs and the definition of the network."""
        if not (listedNamedAdditionalInputs is None):
            for index,inputs in listedNamedAdditionalInputs:
                self.nodes[index].setAdditionalInputs(inputs)
        for index,value in listedInputs:
            self.nodes[index].forward(value) #Does not give the opportunity to feed in multiple inputs at the first nodes
        for node in range(self.edges.numberOfNodes):
            parents=self.edges.parents(node)
            parentsOutputs=[]
            for parent in parents:
                parentsOutputs.extend(self.nodes[parent].outputs.values()) #Does not really maintain ordering
            if parentsOutputs:
                self.nodes[node].forward(parentsOutputs)
    
    def trim(self,outputNode,parameterNode):
        """Gives the node numbers that are parents of outputNode and children
        of parameterNode. Those two nodes included."""
        children=self.edges.DAGFurtherChildren(parameterNode)
        parents=self.edges.DAGFurtherParents(outputNode)
        intersect=np.intersect1d(children,parents)
        return np.append(intersect,np.array([outputNode,parameterNode]))
    
    def getGrad(self,node,parameterNode,parameterName,trimmedNodes):
        """Gets the gradient of the output of node with respect to parameterName
        in paramterNode. Trimmed nodes is the list of nodes potentially to be
        taken into account."""
        self.nodes[node].backward()
        if node==parameterNode:
            if parameterName in self.nodes[parameterNode].parameters:
                return self.nodes[parameterNode].parameters[parameterName].gradient
            elif parameterName in self.nodes[parameterNode].inputs: 
                return self.nodes[parameterNode].inputs[parameterName].gradient
            else:
                raise Exception('Parameter or input not present.')
        else:
            G=0
            for index,parent in enumerate(self.edges.parents(node)):
                if parent in trimmedNodes:
                    gradSelf=list(self.nodes[node].inputs.values())[index].gradient
                    gradNext=self.getGrad(parent,parameterNode,parameterName,trimmedNodes)
                    G+=tensorMult(gradSelf,gradNext) #Maybe better to switch to lists instead of ordered dicts?
            return G
    
    def backward(self,outputNode,parameterNode,parameterName):
        """Performs back propagation on the output of outputNode with 
        respect to paramterName in paramterNode."""
        trimmedNodes=self.trim(outputNode, parameterNode)
        return self.getGrad(outputNode,parameterNode,parameterName,trimmedNodes)
    
    def longInput(self, inputs):
        """Constructs list of [node, vector] out of [node, matrix] list. Rows
        of the matrix are transformed to column vectors."""
        B=inputs[0][1].shape[0]
        n1, X1=inputs[0]
        longInput=[]
        for i in range(B):
            longInput.append([[n1, X1[i][:,None]]])
        if len(inputs)>1:
            for n, X in inputs[1:]:
                for i in range(B):
                    longInput[i].append([n, X[i][:,None]])
        return longInput
    
    def longAddInput(self, addInputs):
        """Constructs list of [node, [name, vector]] out of [node, [name, matrix]]
        list. Rows of the matrix are transformed to columnvectors."""
        if addInputs is None:
            return None
        B=addInputs[0][1][0][1].shape[0]
        longAddInput=[]
        n1=addInputs[0][0]
        input1=addInputs[0][1]
        longInput=self.longInput(input1)
        for i in range(B):
            longAddInput.append([[n1,longInput[i]]])
        if len(addInputs)>1:
            for n, l in addInputs[1:]:
                longInput=self.longInput(l)
                for i in range(B):
                    longAddInput[i].append([n, longInput[i]])
        return longAddInput
                       
    def updateMiniBatch(self, inputs, lr, outputNodeNr, addInputs=None):
        """Updates all parameters with learning rate lr and gradient of output
        node. Inputs and additional inputs have matrix with examples in rows."""
        longInput=self.longInput(inputs)
        longAddInput=self.longAddInput(addInputs)
        self.updateAll(longInput, lr, outputNodeNr, longAddInput)
        
    def loss(self, inputs, outputNodeNr, outputName, addInputs=None):
        """Caluclates the average loss over a batch."""
        out=0
        for inp, addInp in zip(inputs, addInputs):
            self.forward(inp, addInp)
            out+=self.nodes[outputNodeNr].outputs[outputName]
        return out/len(inputs)
    
    def batchLoss(self, inputs, outputNodeNr, outputName='output', addInputs=None):
        """Calculates average output of ouput node over a batch. Inputs and
        additional inputs have matrix with examples in rows."""
        longInput=self.longInput(inputs)
        longAddInput=self.longAddInput(addInputs)
        return self.loss(longInput, outputNodeNr, outputName, longAddInput)

    def longAndShuffle(self, inputs, addInputs=None):
        longInputs=self.longInput(inputs)
        T=len(longInputs)
        order=np.arange(T)
        self.rng.shuffle(order)
        longInputs=[longInputs[i] for i in order]
        if not (addInputs is None):
            longAddInputs=self.longAddInput(addInputs)
            longAddInputs=[longAddInputs[i] for i in order]
        else:
            longAddInputs=addInputs
        return longInputs, longAddInputs
    
    def gradT(self, inputs, outputNodeNr, addInputs=None):
        """Estimate the transposed gradients and store them in a dictionary."""
        G={}
        if not (addInputs is None):
            for nodeNr in range(len(self.nodes)):
                for parameterName in self.nodes[nodeNr].parameters.keys():
                    numberName=str(nodeNr)+parameterName
                    G[numberName]=0
                    for inp, addInp in zip(inputs, addInputs):
                        self.forward(inp, addInp)
                        G[numberName]+=self.backward(outputNodeNr, nodeNr, parameterName)
                    G[numberName]=(G[numberName]/len(inputs)).T
        else:
            for nodeNr in range(len(self.nodes)):
                for parameterName in self.nodes[nodeNr].parameters.keys():
                    numberName=str(nodeNr)+parameterName
                    G[numberName]=0
                    for inp in inputs:
                        self.forward(inp)
                        G[numberName]+=self.backward(outputNodeNr, nodeNr, parameterName)
                    G[numberName]=(G[numberName]/len(inputs)).T
        return G
    
    def updateM(self, oldParams, GTnew, GTold, M):
        """Updates the approximation of the Hessian as used in BFGS. A small 
        constant is added in the division for numerical stability."""
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                h=np.reshape(self.nodes[nodeNr].parameters[parameterName].value-oldParams[numberName], (-1,1), 'F')
                q=np.reshape(GTnew[numberName]-GTold[numberName], (-1,1), 'F')
                M[numberName]=M[numberName]+(1+(np.dot(np.dot(q.T,M[numberName]),q))/(
                    np.dot(h.T,q)+0.000001))*(np.dot(h,h.T))/(np.dot(h.T,q)+0.000001)-1/(np.dot(h.T,q)+0.000001)*(
                    np.dot(np.dot(h,q.T),M[numberName])+np.dot(np.dot(M[numberName],
                    q), h.T)) #Small constant is added for numerical stability
        return M
    
    def updateMVec(self, oldParams, GTnew, GTold, M):
        """Updates the approximation of the Hessian as used in BFGS. A small 
        constant is added in the division for numerical stability. Calculations
        not based on dictionaries, but on a single large M and vecs."""
        newParams={}
        for nodeNr in range(len(self.nodes)):
            for parameterName, vg in self.nodes[nodeNr].parameters.items():
                newParams[str(nodeNr)+parameterName]=vg.value
        h=self.toVec(newParams)-oldParams
        q=GTnew-GTold
        M=M+(1+(np.dot(np.dot(q.T,M),q))/(np.dot(h.T,q)+0.000001))*(
            np.dot(h,h.T))/(np.dot(h.T,q)+0.000001)-1/(np.dot(h.T,q)+0.000001)*(
            np.dot(np.dot(h,q.T),M)+np.dot(np.dot(M,q), h.T)) #Small constant is added for numerical stability
        return M
    
    def setParameters(self, p):
        """Sets the parameters according to the ones given in dictionary p."""
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                self.nodes[nodeNr].parameters[parameterName].value=p[str(nodeNr)+parameterName]
    
    def storeParameters(self):
        """Stores the current parameters in a dictionary."""
        p={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                p[str(nodeNr)+parameterName]=self.nodes[nodeNr].parameters[parameterName].value
        return p
                
    def parameterDistance(self, a, b=None):
        """Returns the Euclidian distance between the parameters in dictionaries
        a and b. Or if b is None, the current parameters are used."""
        d=0
        if b:
            for nodeNr in range(len(self.nodes)):
                for parameterName in self.nodes[nodeNr].parameters.keys():
                   d+=np.sum((a[str(nodeNr)+parameterName]-b[str(nodeNr)+parameterName])**2)
        else:
            for nodeNr in range(len(self.nodes)):
                for parameterName in self.nodes[nodeNr].parameters.keys():
                   d+=np.sum((self.nodes[nodeNr].parameters[parameterName].value-a[str(nodeNr)+parameterName])**2)
        return np.sqrt(d)
    
    def goldenSectionParameters(self, p, inputs, outputNodeNr, outputName, addInputs=None, tol=1e-5):
        """Performs golden section search to find a minimum on the line between
        the current parameters and those given by the dictionary p."""
        ratio=(np.sqrt(5)+1)/2
        pOld={}
        p2={}
        c={}
        d={}
        res={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                a=self.nodes[nodeNr].parameters[parameterName].value
                pOld[numberName]=a
                p2[numberName]=a
                b=p[numberName]
                c[numberName]=b-(b-a)/ratio
                d[numberName]=a+(b-a)/ratio
        while(self.parameterDistance(p)>tol):
            self.setParameters(c)
            fc=self.loss(inputs, outputNodeNr, outputName, addInputs)
            self.setParameters(d)
            fd=self.loss(inputs, outputNodeNr, outputName, addInputs)
            if fc<fd:
                p=d
            else:
                p2=c
            for nodeNr in range(len(self.nodes)):
                for parameterName in self.nodes[nodeNr].parameters.keys():
                    numberName=str(nodeNr)+parameterName
                    a=p2[numberName]
                    b=p[numberName]
                    c[numberName]=b-(b-a)/ratio
                    d[numberName]=a+(b-a)/ratio
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                res[numberName]=(p[numberName]-p2[numberName])/2
        self.setParameters(pOld)
        return res      
    
    def toVec(self, dictionary):
        """Transforms the given dictionary to a vec."""
        if not isinstance(dictionary, dict):
            raise TypeError('Given argument is not a dictionary.')
        res=np.zeros((1,1))
        for value in dictionary.values():
            vec=np.reshape(value, (-1,1), 'F')
            res=np.vstack((res,vec))
        return res[1:,:]

    def toDict(self, vec):
        """Transforms the given vec to dictionary of parameters."""
        d={}
        for nodeNr in range(len(self.nodes)):
            for parameterName,parameter in self.nodes[nodeNr].parameters.items():
                values=vec[:np.size(parameter.value),:]
                vec=vec[np.size(parameter.value):,:]
                d[str(nodeNr)+parameterName]=np.reshape(values,np.shape(parameter.value), 'F')
        return d        
    
    def updateAll(self, inputs, lr, outputNodeNr, addInputs=None):
        """Updates all parameters with learning rate lr and gradient of the 
        output of the output node."""
        GT=self.gradT(inputs, outputNodeNr, addInputs)
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                self.nodes[nodeNr].parameters[parameterName].value-=GT[str(nodeNr)+parameterName]*lr
                    
    def updateAllMomentum(self, inputs, lr, outputNodeNr, v, addInputs=None, a=0.5):
        """Updates all parameters with learning rate lr and gradient of the 
        output of the output node. v is velocity. a is momentum parameter."""
        GT=self.gradT(inputs, outputNodeNr, addInputs)    
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=a*v[str(nodeNr)+parameterName]-GT[str(nodeNr)+parameterName]*lr
                self.nodes[nodeNr].parameters[parameterName].value+=v[str(nodeNr)+parameterName]
        return v
    
    def updateAllNesterov(self, inputs, lr, outputNodeNr, v, addInputs=None, a=0.5):
        """Updates all parameters with learning rate lr and gradient of the 
        output of the output node with Nesterov momentum. v is velocity. a is 
        momentum parameter."""
        oldParams={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                oldParams[str(nodeNr)+parameterName]=self.nodes[nodeNr].parameters[parameterName].value
                self.nodes[nodeNr].parameters[parameterName].value+=a*v[str(nodeNr)+parameterName]
        GT=self.gradT(inputs, outputNodeNr, addInputs)    
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                v[numberName]=a*v[numberName]-GT[numberName]*lr
                self.nodes[nodeNr].parameters[parameterName].value=oldParams[numberName]+v[numberName]
                return v
    
    def updateAllRMSPropNesterov(self, inputs, lr, outputNodeNr, v, r, addInputs=None, a=0.5, rho=0.9):
        """Updates all parameters with learning rate lr and gradient of the 
        output of the output node with RMSProp with Nesterov momentum. v is
        velocity. a is momentum parameter. r is accumulated gradient. rho is
        the parameter governing the acumulation."""
        oldParams={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                oldParams[str(nodeNr)+parameterName]=self.nodes[nodeNr].parameters[parameterName].value
                self.nodes[nodeNr].parameters[parameterName].value+=a*v[str(nodeNr)+parameterName]
        GT=self.gradT(inputs, outputNodeNr, addInputs)    
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                r[numberName]=rho*r[numberName]+(1-rho)*(GT[numberName])**2
                v[numberName]=a*v[numberName]-GT[numberName]*lr/np.sqrt(r[numberName]+0.000001)
                self.nodes[nodeNr].parameters[parameterName].value=oldParams[numberName]+v[numberName]
        return v,r
    
    def updateAllBFGS(self, inputs, outputNodeNr, M, GTvec, addInputs=None):
        """Updates all parameters with learning rate lr and gradient of the 
        output of the output node with BFGS without line search. M is 
        dictionary of approximated Hessians. G is dictionary of previous gradients."""
        oldParams={}
        GTvecnew={}
        #Store old paramters and if G empty calculate gradients
        if not GTvec:
            GTvec={key:np.reshape(value, (-1,1), 'F') for (key,value) in self.gradT(inputs, outputNodeNr, addInputs).items()}
        oldParams=self.storeParameters()
        #Update parameters            
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                update=np.dot(M[numberName],GTvec[numberName])
                update=np.reshape(update, np.shape(self.nodes[nodeNr].parameters[parameterName].value), 'F')
                self.nodes[nodeNr].parameters[parameterName].value-=update
        #Calculate new gradients
        GTvecnew={key:np.reshape(value, (-1,1), 'F') for (key, value) in self.gradT(inputs, outputNodeNr, addInputs).items()}
        #Update M
        M=self.updateM(oldParams, GTvecnew, GTvec, M)
        return M, GTvecnew
    
    def updateAllBFGSls(self, inputs, outputNodeNr, M, GTvec, addInputs=None, tol=1e-5, outputName='output'):
        """Updates all parameters with BFGS with line search. M is 
        dictionary of approximated Hessians. GTvec is dictionary of vec of 
        transposed previous gradients."""
        oldParams={}
        GTvecnew={}
        update={}
        #Store old paramters and if G empty calculate gradients
        if not GTvec:
            GTvec={key:np.reshape(value, (-1,1), 'F') for (key,value) in self.gradT(inputs, outputNodeNr, addInputs).items()}
        oldParams=self.storeParameters()
        #Update parameters            
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                numberName=str(nodeNr)+parameterName
                u=np.dot(M[numberName],GTvec[numberName])
                u=np.reshape(u, np.shape(self.nodes[nodeNr].parameters[parameterName].value), 'F')
                update[numberName]=self.nodes[nodeNr].parameters[parameterName].value-u
        update=self.goldenSectionParameters(update, inputs, outputNodeNr, outputName, addInputs, tol)
        self.setParameters(update)
        #Calculate new gradients
        GTvecnew={key:np.reshape(value, (-1,1), 'F') for (key, value) in self.gradT(inputs, outputNodeNr, addInputs).items()}
        #Update M
        M=self.updateM(oldParams, GTvecnew, GTvec, M)
        return M, GTvecnew
    
    def updateAllBFGSTot(self, inputs, outputNodeNr, M, GTvec, addInputs=None, tol=1e-5, outputName='output'):
        """Updates all parameters simultaneously with BFGS with line search. M
        is big matrix of approximated Hessian. GTvec is dictionary of vec of 
        transposed previous gradients."""
        oldParams={}
        #Store old paramters and if G empty calculate gradients
        if GTvec is None:
            GTvec=self.toVec(self.gradT(inputs, outputNodeNr, addInputs))
        oldParams=self.storeParameters()
        
        #Update parameters  
        update=np.dot(M, GTvec)
        update=self.toDict(update)
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                update[str(nodeNr)+parameterName]=self.nodes[nodeNr].parameters[parameterName].value-update[str(nodeNr)+parameterName]
        update=self.goldenSectionParameters(update, inputs, outputNodeNr, outputName, addInputs, tol)
        self.setParameters(update)
          
        #Calculate new gradients
        GTvecnew=self.toVec(self.gradT(inputs, outputNodeNr, addInputs))
        
        #Update M
        M=self.updateMVec(self.toVec(oldParams), GTvecnew, GTvec, M)
        return M, GTvecnew
        
    def SGDMomentum(self, inputs, lr, outputNodeNr, B, epochs, outputName='output', addInputs=None, a=0.5):
        """Performs stochastic gradient descent with batch size B for the 
        specified number of epochs. a is the momentum paramter. Momentums are
        stored in dictionairy."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        v={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v=self.updateAllMomentum(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, longShuffAddInputs[B*j:B*(j+1)],a)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                v=self.updateAllMomentum(longShuffInputs[B*j:], lr, outputNodeNr, v, longShuffAddInputs[B*j:], a)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v=self.updateAllMomentum(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, a)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                v=self.updateAllMomentum(longShuffInputs[B*j:], lr, outputNodeNr, v, a)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses 

    def SGDNesterov(self, inputs, lr, outputNodeNr, B, epochs, outputName='output', addInputs=None, a=0.5):
        """Performs stochastic gradient descent with batch size B for the 
        specified number of epochs. a is the momentum paramter. Momentums are
        stored in dictionairy. Nesterov momentum."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        v={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v=self.updateAllNesterov(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, longShuffAddInputs[B*j:B*(j+1)],a)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                v=self.updateAllNesterov(longShuffInputs[B*j:], lr, outputNodeNr, v, longShuffAddInputs[B*j:], a)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v=self.updateAllNesterov(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, a)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                v=self.updateAllNesterov(longShuffInputs[B*j:], lr, outputNodeNr, v, a)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses 

    def RMSPropNesterov(self, inputs, lr, outputNodeNr, B, epochs, outputName='output', addInputs=None, a=0.5, rho=0.9):
        """Performs stochastic gradient descent with batch size B for the 
        specified number of epochs. a is the momentum paramter. Momentums are
        stored in dictionairy. Nesterov momentum."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        v={}
        r={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
                r[str(nodeNr)+parameterName]=0
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v,r=self.updateAllRMSPropNesterov(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, r, longShuffAddInputs[B*j:B*(j+1)],a, rho)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                v,r=self.updateAllRMSPropNesterov(longShuffInputs[B*j:], lr, outputNodeNr, v, r, longShuffAddInputs[B*j:], a, rho)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    v,r=self.updateAllRMSPropNesterov(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, v, r, a, rho)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                v,r=self.updateAllRMSPropNesterov(longShuffInputs[B*j:], lr, outputNodeNr, v, r, a, rho)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses 

    def SGD(self, inputs, lr, outputNodeNr, B, epochs, outputName='output', addInputs=None):
        """Performs stochastic gradient descent with batch size B for the 
        specified number of epochs."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    self.updateAll(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr, longShuffAddInputs[B*j:B*(j+1)])
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                self.updateAll(longShuffInputs[B*j:], lr, outputNodeNr, longShuffAddInputs[B*j:])
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    self.updateAll(longShuffInputs[B*j:B*(j+1)], lr, outputNodeNr)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                self.updateAll(longShuffInputs[B*j:], lr, outputNodeNr)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses
    
    def BFGSnl(self, inputs, outputNodeNr, B, epochs, outputName='output', addInputs=None):
        """Performs BFGS without line search with batch size B for the 
        specified number of epochs."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        M={}
        GTvec={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                M[str(nodeNr)+parameterName]=np.eye(np.size(self.nodes[nodeNr].parameters[parameterName].value))
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GTvec=self.updateAllBFGS(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:B*(j+1)])
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                M, GTvec=self.updateAllBFGS(longShuffInputs[B*j:], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:])
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GT=self.updateAllBFGS(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                M, GTvec=self.updateAllBFGS(longShuffInputs[B*j:], outputNodeNr, M, GTvec)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses
    
    def BFGSTot(self, inputs, outputNodeNr, B, epochs, outputName='output', addInputs=None):
        """Performs BFGS without line search with batch size B for the 
        specified number of epochs."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        GTvec=None
        size=0
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                size+=np.size(self.nodes[nodeNr].parameters[parameterName].value)
        M=np.eye(size)

        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GTvec=self.updateAllBFGSTot(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:B*(j+1)])
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                M, GTvec=self.updateAllBFGSTot(longShuffInputs[B*j:], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:])
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GT=self.updateAllBFGSTot(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                M, GTvec=self.updateAllBFGSTot(longShuffInputs[B*j:], outputNodeNr, M, GTvec)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses
    
    def BFGS(self, inputs, outputNodeNr, B, epochs, outputName='output', addInputs=None, tol=1e-5):
        """Performs BFGS with line search with batch size B for the specified 
        number of epochs. Tol gives tolerance in line search."""
        T=inputs[0][1].shape[0]
        iters=int(np.ceil(T/B))
        losses=np.zeros((epochs,iters))
        M={}
        GTvec={}
        #Initialise M to identity
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                M[str(nodeNr)+parameterName]=np.eye(np.size(self.nodes[nodeNr].parameters[parameterName].value))
        if not (addInputs is None):
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GTvec=self.updateAllBFGSls(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:B*(j+1)], tol, outputName)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName, longShuffAddInputs[B*j:B*(j+1)])
                j=iters-1
                M, GTvec=self.updateAllBFGSls(longShuffInputs[B*j:], outputNodeNr, M, GTvec, longShuffAddInputs[B*j:], tol, outputName)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName, longShuffAddInputs[B*j:])
        else:
            for i in range(epochs):
                longShuffInputs,longShuffAddInputs=self.longAndShuffle(inputs, addInputs)
                for j in range(iters-1):
                    M, GT=self.updateAllBFGSls(longShuffInputs[B*j:B*(j+1)], outputNodeNr, M, GTvec, tol=tol, outputName=outputName)
                    losses[i,j]=self.loss(longShuffInputs[B*j:B*(j+1)], outputNodeNr, outputName)
                j=iters-1
                M, GTvec=self.updateAllBFGSls(longShuffInputs[B*j:], outputNodeNr, M, GTvec, tol=tol, outputName=outputName)
                losses[i,j]=self.loss(longShuffInputs[B*j:], outputNodeNr, outputName)
        return losses
    
    def sequentialOut(self, inputs, nodeNr, outputName='output', addInputs=None):
        """Calculates the ouput of nodeNr for the different examples presented 
        in inputs."""
        longInput=self.longInput(inputs)
        longAddInput=self.longAddInput(addInputs)
        self.forward(longInput[0], longAddInput[0])
        res=np.zeros((len(longInput),len(self.nodes[nodeNr].outputs[outputName])))
        res[0]=np.squeeze(self.nodes[nodeNr].outputs[outputName])
        for i,(inp, addInp) in enumerate(zip(longInput[1:], longAddInput[1:])):
            self.forward(inp, addInp)
            res[i+1]=np.squeeze(self.nodes[nodeNr].outputs[outputName])
        return res
    
    def batchIter(self, B, *args):
        """Creates a generator object that yields batches of size B of the
        arguments given. When the arguments are exhausted, they are shuffeled
        and iteration starts again."""
        size=len(args[0])
        order=np.arange(size)
        for arg in args:
            if len(arg) != size:
                raise Exception('Lists must be of same length.')
        while True:
            self.rng.shuffle(order)
            args=[[arg[i] for i in order] for arg in args]
            for i in range(0, len(args[0]), B):
                yield [arg[i:i+B] for arg in args]
                
    def iterateGenerator(self, trainData, B, iterations, validData=None):
        """Creates a batchIterator and runs calls next iterations times on it."""
        inputs,addInputs=trainData
        if validData is None:
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)
            else:
                gen=self.batchIter(B, lsInputs)
        else:
            val, addVal=validData
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            lsVal,lsAddVal=self.longAndShuffle(val, addVal)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)
        for i in range(iterations):
            next(gen)
            
    def trainRMSPropStepSize(self, trainData, flr, fa, frho, p, step, outputNodeNr,
                             B, outputName='output', validData=None, maxUpdates=5000,
                             name='0bias', onePerOne=True):
        """Trains the network. If no validation data is passed, it will continue
        training until the loss has not decreased for p evaluations. Or when the
        maximum amount of updates is reached. If validation data is passed 
        training continues untill the loss over the validation set evaluated 
        every step iteration has not decreased for p iterations.
        lrf, af, rhof are the learning rate, momentum parameter and rho functions.
        Prints step sizes of the updates"""        
        inputs,addInputs=trainData
        v={}
        r={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
                r[str(nodeNr)+parameterName]=0
        k=0
        j=0
        kOpt=0
        losses=[]
        vs=[]
        rs=[]
        minLoss=np.inf
        paramOpt=self.storeParameters()
        if validData is None:
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)    
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, addInp, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName, lsAddInputs))
                    vs.append(v[name])
                    rs.append(r[name])
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                    if onePerOne:
                        print('Loss', losses[-1])
                        print('V', v[name])
                        print('R', r[name])
                        print('Next?')
                        input()
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)    
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName))
                    vs.append(v[name])
                    rs.append(r[name])
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1 
                    if onePerOne:
                        print('Loss', losses[-1])
                        print('V', v[name])
                        print('R', r[name])
                        print('Next?')
                        input()
        else: #With valid data
            initialParam=self.storeParameters()
            val, addVal=validData
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            lsVal,lsAddVal=self.longAndShuffle(val, addVal)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, addInp, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName, lsAddVal))
                    vs.append(v[name])
                    rs.append(r[name])
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                    if onePerOne:
                        print('Loss', losses[-1])
                        print('V', v[name])
                        print('R', r[name])
                        print('Next?')
                        input()
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                        r[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal, lsAddInputs+lsAddVal)
                for i in range(step*kOpt):
                    inpComb, addInpComb=next(genComb)
                    v,r=self.updateAllRMSPropNesterov(inpComb, flr(i), outputNodeNr, v, r, addInpComb, fa(i), frho(i))
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName))
                    vs.append(v[name])
                    rs.append(r[name])
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                    if onePerOne:
                        print('Loss', losses[-1])
                        print('V', v[name])
                        print('R', r[name])
                        print('Next?')
                        input()
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                        r[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal)
                for i in range (step*kOpt):
                    inpComb=next(genComb)
                    v,r=self.updateAllRMSPropNesterov(inpComb, flr(i), outputNodeNr, v, r, fa(i), frho(i)) 
        return np.array(losses), kOpt, paramOpt, minLoss, vs, rs
    
    
    def trainRMSProp(self, trainData, flr, fa, frho, p, step, outputNodeNr, B, outputName='output', validData=None, maxUpdates=5000):
        """Trains the network. If no validation data is passed, it will continue
        training until the loss has not decreased for p evaluations. Or when the
        maximum amount of updates is reached. If validation data is passed 
        training continues untill the loss over the validation set evaluated 
        every step iteration has not decreased for p iterations.
        lrf, af, rhof are the learning rate, momentum parameter and rho functions."""        
        inputs,addInputs=trainData
        v={}
        r={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
                r[str(nodeNr)+parameterName]=0
        k=0
        j=0
        kOpt=0
        losses=[]
        minLoss=np.inf
        paramOpt=self.storeParameters()
        if validData is None:
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)    
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, addInp, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName, lsAddInputs))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)    
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1    
        else: #With valid data
            initialParam=self.storeParameters()
            val, addVal=validData
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            lsVal,lsAddVal=self.longAndShuffle(val, addVal)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, addInp, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName, lsAddVal))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                        r[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal, lsAddInputs+lsAddVal)
                for i in range(step*kOpt):
                    inpComb, addInpComb=next(genComb)
                    v,r=self.updateAllRMSPropNesterov(inpComb, flr(i), outputNodeNr, v, r, addInpComb, fa(i), frho(i))
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp=next(gen)
                        v,r=self.updateAllRMSPropNesterov(inp, flr(k), outputNodeNr, v, r, fa(k), frho(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                        r[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal)
                for i in range (step*kOpt):
                    inpComb=next(genComb)
                    v,r=self.updateAllRMSPropNesterov(inpComb, flr(i), outputNodeNr, v, r, fa(i), frho(i)) 
        return np.array(losses), kOpt, paramOpt, minLoss
    
    def trainNesterov(self, trainData, flr, fa, p, step, outputNodeNr, B, outputName='output', validData=None, maxUpdates=5000):
        """Trains the network. If no validation data is passed, it will continue
        training until the loss has not decreased for p evaluations or max updates
        is reached. If validation data is passed training continues untill the
        loss over the validation set evaluated every step iteration has not 
        decreased for p iterations.
        lrf, af, rhof are the learning rate, momentum parameter and rho functions."""
        inputs,addInputs=trainData
        v={}
        for nodeNr in range(len(self.nodes)):
            for parameterName in self.nodes[nodeNr].parameters.keys():
                v[str(nodeNr)+parameterName]=0
        k=0
        j=0
        kOpt=0
        losses=[]
        minLoss=np.inf
        paramOpt=self.storeParameters()
        if validData is None:
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)    
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v=self.updateAllNesterov(inp, flr(k), outputNodeNr, v, addInp, fa(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName, lsAddInputs))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)    
                while j<p:
                    for i in range(step):
                        inp=next(gen)
                        v=self.updateAllNesterov(inp, flr(k), outputNodeNr, v, a=fa(k))
                        k+=1
                    losses.append(self.loss(lsInputs, outputNodeNr, outputName))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1    
        else:
            initialParam=self.storeParameters()
            val, addVal=validData
            lsInputs,lsAddInputs=self.longAndShuffle(inputs, addInputs)
            lsVal,lsAddVal=self.longAndShuffle(val, addVal)
            if not (addInputs is None):
                gen=self.batchIter(B, lsInputs, lsAddInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp, addInp=next(gen)
                        v=self.updateAllNesterov(inp, flr(k), outputNodeNr, v, addInp, fa(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName, lsAddVal))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal, lsAddInputs+lsAddVal)
                for i in range(step*kOpt):
                    inpComb, addInpComb=next(genComb)
                    v=self.updateAllNesterov(inpComb, flr(i), outputNodeNr, v, addInp, fa(i))
            else: #No add inputs
                gen=self.batchIter(B, lsInputs)
                while j<p and k<maxUpdates:
                    for i in range(step):
                        inp=next(gen)
                        v=self.updateAllNesterov(inp, flr(k), outputNodeNr, v, a=fa(k))
                        k+=1
                    losses.append(self.loss(lsVal, outputNodeNr, outputName))
                    if losses[-1]<minLoss:
                        minLoss=losses[-1]
                        j=0
                        kOpt=k
                        paramOpt=self.storeParameters()
                    elif np.isnan(losses[-1]) or np.isinf(losses[-1]):
                        break
                    else:
                        j+=1
                #Retrain with all the data
                for nodeNr in range(len(self.nodes)):
                    for parameterName in self.nodes[nodeNr].parameters.keys():
                        v[str(nodeNr)+parameterName]=0
                self.setParameters(initialParam)
                genComb=self.batchIter(B, lsInputs+lsVal)
                for i in range (step*kOpt):
                    inpComb=next(genComb)
                    v=self.updateAllNesterov(inpComb, flr(i), outputNodeNr, v, addInp, a=fa(i))   
        return np.array(losses), kOpt, paramOpt, minLoss
                              
class FeedForward(Graph):
    """Defines a fully connected feed forward ReLU network. Initialised with
    random normals for weights and zero bias.
    
    Layers are the number of layers. Last layer only affine transformation.
    Widths is layers+1 list of widths. First entry is input dimension."""
    def __init__(self, layers, widths, rng=None):
        super().__init__(rng)
        edgeList=list(zip(range(0,layers-1),range(1,layers)))
        self.edges=Edges(edgeList)
        for i in range(layers-1):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),0.1)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,1,(widths[-1],widths[-2]))
        b=np.zeros((widths[-1],1))
        self.nodes.append(AffineNode(W,b))
        
class FeedForwardLoss(Graph): #Note, bias is changed
    """Defines a fully connected feed forward ReLU network with loss function.
    Initialised with random normals for weights and unit bias for ReLU and
    affine nodes.
    
    Layers are the number of layers. Last layer only affine transformation and
    always has 3 dimensional output.
    Widths is layers list of widths. First entry is input dimension."""
    def __init__(self, widths, lossBias=np.full((3,1),0.1), rng=None, ReLUBias=0.1):
        super().__init__(rng)
        if lossBias.shape!=(3,1):
            raise Exception('Bias in last node does not have correct dimensions. Needs (3,1).')
        layers=len(widths)-1
        edgeList=list(zip(range(0,layers+1),range(1,layers+2)))
        self.edges=Edges(edgeList)
        for i in range(layers):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),ReLUBias)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,np.sqrt(2/widths[-1]),(3,widths[-1]))
        b=lossBias
        self.nodes.append(AffineNode(W,b))
        self.nodes.append(AverageNegativeLogNormalNode())
        
class FeedForwardLossLogSigma(Graph): #Note, bias has changed
    """Defines a fully connected feed forward ReLU network with loss function.
    Assumes log simga is returned by the network.
    Initialised with random normals for weights and unit bias for ReLU and
    affine nodes.
    
    Widths is layers list of widths. First entry is input dimension. Then
    follow the hidden layers. Last hidden layer always has output dimension
    three. Which is then fed into the log sigma loss function."""
    def __init__(self, widths, lossBias=np.full((3,1),0.1), rng=None, ReLUBias=0.1):
        super().__init__(rng)
        if lossBias.shape!=(3,1):
            raise Exception('Bias in last node does not have correct dimensions. Needs (3,1).')
        layers=len(widths)-1
        edgeList=list(zip(range(0,layers+1),range(1,layers+2)))
        self.edges=Edges(edgeList)
        for i in range(layers):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),ReLUBias)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,np.sqrt(2/widths[-1]),(3,widths[-1]))
        b=lossBias
        self.nodes.append(AffineNode(W,b))
        self.nodes.append(AvNegLogNormalNodeLogSigma())
        
class FeedForwardLogSigmaExpLoss(Graph): #Note, bias has changed
    """Defines a fully connected feed forward ReLU network with loss function.
    Assumes log simga is returned by the network. Then exponentiates it and feeds
    it through loss for sigma.
    Initialised with random normals for weights and unit bias for ReLU and
    affine nodes.
    
    Widths is layers list of widths. First entry is input dimension. Then
    follow the hidden layers. Last hidden layer always has output dimension
    three. Which is then fed into the log sigma loss function."""
    def __init__(self, widths, lossBias=np.full((3,1),0.1), rng=None, ReLUBias=0.1):
        super().__init__(rng)
        if lossBias.shape!=(3,1):
            raise Exception('Bias in last node does not have correct dimensions. Needs (3,1).')
        layers=len(widths)-1
        edgeList=list(zip(range(0,layers+2),range(1,layers+3)))
        self.edges=Edges(edgeList)
        for i in range(layers):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),ReLUBias)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,np.sqrt(2/widths[-1]),(3,widths[-1]))
        b=lossBias
        self.nodes.append(AffineNode(W,b))
        self.nodes.append(ExpNode([2]))
        self.nodes.append(AverageNegativeLogNormalNode())
        
class FeedForwardT(Graph): #Note, bias has changed
    """Defines a fully connected feed forward ReLU network with loss function
    from a t distriubtion. Assumes log simga and log nu are returned by the network.
    Initialised with random normals for weights and unit bias for ReLU and
    affine nodes.
    
    Widths is layers list of widths. First entry is input dimension. Then
    follow the hidden layers. Last hidden layer always has output dimension
    three. Which is then fed into the log sigma, log nu t-loss function."""
    def __init__(self, widths, lossBias=np.array([[0.1],[0.1],[0],[1.7]]), rng=None, ReLUBias=0.1):
        super().__init__(rng)
        if lossBias.shape!=(4,1):
            raise Exception('Bias in last node does not have correct dimensions. Needs (4,1).')
        layers=len(widths)-1
        edgeList=list(zip(range(0,layers+1),range(1,layers+2)))
        self.edges=Edges(edgeList)
        for i in range(layers):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),ReLUBias)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,np.sqrt(2/widths[-1]),(4,widths[-1]))
        b=lossBias
        self.nodes.append(AffineNode(W,b))
        self.nodes.append(AvNegLogtNodeLogSigmaLogNu())

class FeedForwardSquareLoss(Graph):
    """Defines a fully connected feed forward ReLU network. Initialised with
    random normals for weights and zero bias.
    
    Layers are the number of layers. Last layer only affine transformation to
    scalar, which is then passed on to quadratic loss.
    Widths is layers list of widths. First entry is input dimension."""
    def __init__(self, layers, widths, rng=None):
        super().__init__(rng)
        edgeList=list(zip(range(0,layers),range(1,layers+1)))
        self.edges=Edges(edgeList)
        for i in range(layers-1):
            W=self.rng.normal(0,np.sqrt(2/widths[i]),(widths[i+1],widths[i]))
            b=np.full((widths[i+1],1),0.1)
            self.nodes.append(ReLUNode(W,b))
        W=self.rng.normal(0,np.sqrt(2/widths[-1]),(1,widths[-1]))
        b=0
        self.nodes.append(AffineNode(W,b))
        self.nodes.append(quadraticLossNode())

def generateData(T, d, rng):
    eps=rng.normal(0,1,(T,1))
    x=rng.normal(0,1,(T,d))
    y=np.sum(x,1).reshape((len(x),1))+0*eps
    return y,x

if __name__ == '__main__':
    ### Test code
    seed=28051998
    rng=np.random.default_rng(seed)
    y,x=generateData(10000,2,rng)
    
    widths8=[2,5,5]
    outputNode8=len(widths8)
    batchSize8=25
    epochs8=10
    
    inputX=[[0,x]]
    inputY8=[[outputNode8,[['y', y]]]]
    
    ff8=FeedForwardSquareLoss(outputNode8,widths8,rng)
    res9=ff8.BFGSTot(inputX, outputNode8, batchSize8, epochs8, 'output', inputY8)
    print(np.mean(res9,1))
    #res8=ff8.BFGS(inputX, outputNode8, batchSize8, epochs8, 'output', inputY8)
    #np.mean(res8,1)
    #yHat8=ff8.sequentialOut(inputX, outputNode8-1, 'output', inputY8)
    #(y-yHat8)[:10,:]
    #np.mean(y-yHat8)
    
    