# training notes

## strategies

**fine-tuning**

Take a trained model and expose it to samples it scores poorly on.
Re-evaluate, and re-establish sampling frequencies.

**rejection sampling**

At least at the beginning of training, want to expose the model to
a uniform distribution of sample types (ie static combos).

Let the normalized frequency of soil/veg combinations be a Nd random
variable

`X = (X_1,X_2,X_3)`

Where each component is a discrete random variable

`X_i = (x_{i,0}, x_{i,1}, ... x_{i,K_i}`

Then given a random sample

**curriculum learning**

Goal is to use a-priori knowledge of the difficulty of samples to
inform sampling frequency; introduce simpler samples first.

 1. Define a *scoring function* that determines the "difficulty"
    of each input/output pair.
 2. Define a *pacing function* that determines how quicklcy data is
    exposed to the network.
