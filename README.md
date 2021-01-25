# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. Each memory stores information learned about the dynamical and sensory properties of the environment associated with the corresponding context.

## Dependencies

The COIN model requires installation of the following packages, which improve the efficiency of the code.

- "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. Once downloaded, run install_lightspeed.m to compile the mex files.
- "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh. Once downloaded, run make.m to compile the mex files.
- "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each of the above packages to the search path using 
```
addpath(genpath('directory'))
```
where 'directory' is the full path to the root folder of each package.

## Running the model

The COIN model is implemented as a class in MATLAB. An object of the class can be created by calling
```
obj = COIN;
```
This object has a number of properties that define the model (number of particles, model parameters) and the paradigm to be simulated (perturbations, sensory cues). Additional properties allow the user to specify which variables to plot.

As an example, to simulate a spontaneous recovery paradigm, define a series of perturbations (use NaN to indicate a channel trial):
```
obj.x = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)];
```
and call the infer_COIN method on object obj:
```
[D,w] = obj.infer_COIN;
```
To plot the adaptation:
```
figure
plot(obj.x,'k')
hold on
plot(D.stored.y,'m.')
plot(D.stored.yHat,'c')
legend('perturbation','state feedback','adaptation')
```
We can repeat the simulation multiple times (each based on a different sequence of observation noise) using the property *R*. For example, to run 2 simulations, call
```
obj.R = 2;
[D,w] = obj.infer_COIN;
```
The output of each and each simulation, or run, is assigned a weight (w). In the absence of behavioural adaptation data, all runs are assigned equal weight. If we pass adaptation data via the adaptation property
The more simulations we perform (the greater R is), the greater the computational complexity. If you have access to a computer cluster, you can perform each simulation in parallel, which will speed things up. To do this, specify the maximum number of CPU cores you have access to via the property *maxCores*.

### Parallel Computing

This simulation performed inference based on a single sample of the observation noise, which transforms the perturbation into the state feedback. 
### Performing online inference

### Evaluating the log likelihood

### Inferring internal representations of the COIN model fit to adaptation data

### Plotting internal representations

