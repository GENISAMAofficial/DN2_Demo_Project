DN2 Package Contents

kernels: Directory with OpenCl kernels

DN_GPU.java: Java class. DN class with GPU programming support. GPU is made from OpenCl.

DN2.java: Java class. It is the main point of interaction between the main program and the other classes within this package.

Glial.java: Java class. It implements the pulling of Neurons within the 3D location space. 

HiddenLayer.java: Java class. Manages the interactions between the main program(s) and the  other layers. It receives sensory and motor inputs, send the information to their respective layers for DN2 procedures.

InhibitoryNeuron.java: Java class. Used for Lateral inhibition of the hidden layer neurons. 

MotorLayer.java: Java class. It receives motor input from DN2 class, process the motor input for Hiddenlayer class' computations, and receives HiddenLayer inputs for motor response computation. 
 

Neuron.java: Java class. Does response computations and top-k competition at an individual neuron level.

Pair.java: Java class. Used for Top-k competition of the neurons. It contains a value and an index.

PrimaryHiddenLayer.java: Java class. Manages the interactions between the main program(s) and the  other layers. It receives sensory and motor inputs, send the information to their respective layers for DN2 procedures. All neurons are type 4. 

SensorLayer.java: It receives sensory input from DN1 class, and process the sensory input for HiddenLayer class' computations.
