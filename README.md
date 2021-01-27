# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. Each memory stores information learned about the dynamical and sensory properties of the environment associated with the corresponding context.

## Dependencies

The COIN model requires installation of the following packages, which improve the efficiency of the code:

- "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. Run install_lightspeed.m to compile the mex files.
- "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh. Run make.m to compile the mex files.
- "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each package to MATLAB's search path using 
```
addpath(genpath('directory'))
```
where 'directory' is the full path to the root folder of the package.

## The model

### Running the model

The COIN model is implemented as a class in MATLAB. An object of the class can be created by calling
```
obj = COIN;
```
This object has a number of [properties](#properties) that define the model (e.g. number of particles, model parameters) and the paradigm to be simulated (perturbations, sensory cues). Additional properties allow the user to specify which variables to plot.

To simulate learning with a simple paradigm, first define a series of perturbations (use NaN to indicate a channel trial):
```
obj.x = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)];
```
and then call the run_COIN method on object obj:
```
[S,w] = obj.run_COIN;
```
To plot the adaptation:
```
figure
plot(obj.x,'k')
hold on
plot(S{1}.y,'m.')
plot(S{1}.yHat,'c')
legend('perturbation','state feedback','adaptation')
```
The state feedback is the perturbation plus random observation noise. In general, the actual observation noise that a participant perceives is unknown to us. We can use the property R to run multiple simulations&mdash;each conditioned on a different random sequence of observation noise. For example, to run 2 simulations, call
```
obj.R = 2;
[S,w] = obj.run_COIN;
```
### Plotting internal representations

Use properties to indicate which variables to plot, as well as any additional information such as the values at which to evaluate a distribution at. For example, to plot the distribution of the state of each context and the predicted probabilities, the following properties can be set as
```
obj.xPredPlot = true; % state | context
obj.gridX = linspace(-1.5,1.5,500); % values of the state at which to evaluate state | context
obj.cPredPlot = true; % predicted probabilities
```
Note that these properties must be set *before* running the model so that the relevant variables can be stored for plotting (online inference does not require all variables on all trials to be stored in memory, and so to reduce memory requirements, variables are stored on request).

### Integrating out observation noise
The basic simulation above  have performed inference conditioned on a random sequence of observation noise.
We can repeat the simulation multiple times (each based on a different sequence of observation noise) using the property *R*. For example, to run 2 simulations, call
The output of each and each simulation, or run, is assigned a weight (*w*). In the absence of behavioural adaptation data, all runs are assigned equal weight. If we pass adaptation data via the adaptation property
The more simulations we perform (the greater R is), the greater the computational complexity. If you have access to a computer cluster, you can perform each simulation in parallel, which will speed things up. To do this, specify the maximum number of CPU cores you have access to via the property *maxCores*.

### Parallel Computing

The computational complexity of the COIN model scales linearly wit the number of simulations (obj.R). To reduce runtime, each simulation can be performed in parallel across multiple CPU cores. To engage parallel processing, use the maxCores property to specify the maximum number of CPU cores available for use. The default setting of maxCores is 0, which implements serial processing.

This simulation performed inference based on a single sample of the observation noise, which transforms the perturbation into the state feedback. 
### Performing online inference

### Model fitting

### Storing variables

### Evaluating the log likelihood

### Inferring internal representations of the COIN model fit to adaptation data

## Class properties

