import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

x_height = np.shape(X)[0]
print x_height

x_width = np.shape(X)[1]
print x_width
    
# output dataset            
y = np.array([[0,1,1,2]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# The first "dimension" is how many variables you have. This is very important
syn0 = 2*np.random.random((x_width,x_height)) - 1
syn1 = 2*np.random.random((x_height,x_width)) - 1
syn2 = 2*np.random.random((x_width,x_height)) - 1
syn3 = 2*np.random.random((x_height,x_width)) - 1
syn4 = 2*np.random.random((x_width,1)) - 1
for j in xrange(100000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    l4 = nonlin(np.dot(l3,syn3))
    l5 = nonlin(np.dot(l4,syn4))


    l5_error = y - l5
    l5_delta = l5_error*nonlin(l5, deriv=True)

    l4_error = l5_delta.dot(syn4.T)
    l4_delta = l4_error*nonlin(l4, deriv=True)

    # how much did we miss?
    l3_error = l4_delta.dot(syn3.T)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l3_delta = l3_error*nonlin(l3, deriv=True)

    # do same with l2
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error*nonlin(l2, deriv=True)

    # do same with l1
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    # update weights
    syn4 += l4.T.dot(l5_delta)
    syn3 += l3.T.dot(l4_delta)
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


print "Output After Training:"
print l4
print l5








