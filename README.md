# Neural Network model

## Neuron class
This class defines a single neuron. This can also act as a special purpose neuron with fixed output value.
- Fields
    - `label`: label of this neuron
    - `weight`: weight vector $\vec{w}$. The last element corresponds to the threshold.
    - `function`: response function $f$
    - `fixOutput`: always responds 1 if set true
- Constructors
    1. Given `label`, `weight` and `function`
    2. Given `label`, `function` and the number of inputs `n`. The weights are generated randomly. 
- Methods
    - `response()`: computes the response for given input $\vec{s}$
        - Throws Exception if the size  of the input is not equal to the size of the weight vector.
        - returns $f(\vec{w}\cdot\vec{s})$.
    - `update()`: updates weight vector during learning processes 
        - $\vec{w}\to \vec{w}-\epsilon\vec{s}$
    - `product()`: returns inner product of given two lists of doubles. This method is `static`.

## Layer class
This class defines a single layer class containing neurons.
- Fields
    - `neurons`: list of neurons of this layer
    - `numInput`: the number of inputs to this layer
- Constructors
    1. Given the number of neurons, the number of inputs `numInput`, the response function $f$, and boolean flag to show whether this layer contains a neuron of fixed output.  Weights of neurons are randomly given.
    2. Given the list of neurons.
- Methods
    - `response()`: computes the response of this layer for given input $\vec{s}$
        - Throws Exception if the size of the input is not equal to `numInput`
        - returns $f(\vec{w_i}\cdot\vec{s})$ for each neuron $i$

## Multilayer class
This class defines a multilayer neural network model.
- Fields
    - `layers`: list of layers
    - `derivatives`: list of the derivatives of the response functions, which are used for learning
- Constructor
    1. Given 
        - the number of inputs to the system
        - the list of the numbers of neurons in layers
        - the list of the response functions of layers
        - the list of the derivatives of response functions
        - quantities of each list are ordered from the input side to the output side
    2. Given
        - the number of inputs to the system
        - the list of layers
        - the list of the derivatives of response functions
- Methods
    - `response()`: computes the response of this system for given input $\vec{s}$
    