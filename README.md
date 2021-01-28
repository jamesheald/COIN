# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. Each memory stores information learned about the dynamical and sensory properties of the environment associated with the corresponding context.

The model is described in detail in a [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1) paper.

## Dependencies

The COIN model requires installation of the following packages, which improve the efficiency of the code:

- "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. Run install_lightspeed.m to compile the necessary mex files.
- "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh. Run make.m to compile the necessary mex files.
- "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each package to the MATLAB search path using 
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
This object has a number of [properties](#properties) that define the model (e.g. number of particles, model parameters) and the paradigm (e.g. perturbations, sensory cues).

To simulate learning on a simple paradigm, define a series of perturbations (use NaN to indicate a channel trial):
```
obj.x = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)]; % spontaneous recovery paradigm
```
and run the model by calling the run_COIN method on object obj:
```
[S,w] = obj.run_COIN;
```
This uses a default set of parameters stored in the properties of the object. The adaptation and state feedback are contained in a structure in a cell in S:
```
plot(S{1}.y,'m.')
hold on
plot(S{1}.yHat,'c')
legend('state feedback','adaptation')
```
The state feedback is the perturbation plus random observation noise. In general, the actual observation noise that a participant perceives is unknown to us. The R property can be used to perform multiple runs of the simulation&mdash;each conditioned on a different random sequence of observation noise. For example, to perform 2 runs:
```
obj.R = 2;
[S,w] = obj.run_COIN;
```
Here S is an array with one cell per run and w is vector specifying the relative weight of each run. If the model was fit to adaptation data, each run can be assigned a weight based on how well it explains the data (see [Inferring internal representations fit to adaptation data](#inferring-internal-representations-fit-to-adaptation-data)). In the above example, each run is assigned an equal weight as the adaptation property was not defined. 

### Plotting internal representations

To plot specific variables or distributions, activate the relevant flags in the properties of the class object. If plotting a continuous distribution, also specify the points at which to evaluate the distribution. For example, to plot the distribution of the state of each context and the predicted probabilities:
```
obj.xPredPlot = true; % I want to plot the state | context
obj.gridX = linspace(-1.5,1.5,500); % points to evaluate state | context at
obj.cPredPlot = true; % I want to plot the predicted probabilities
```
These properties must be set before running the model so that the variables needed to generate the plots can be stored. For information on which variables are available for plotting and how to plot them, see [Properties](#properties).

After running the model, call the plot_COIN method on object obj:
```
[P,S] = obj.plot_COIN(S,w);
```
This will generate the desired plots&mdash;a state | context plot and a predicted probabilities plot. The structure P contains the data that is plotted (view the generate_figures method in COIN.m to see how the data in P is plotted). Note that these plots may take some time to generate, as they require contexts in multiple particles and multiple runs to be relabelled on each trial. Once these contexts have been relabelled, the relevant variables or distributions are averaged across particles and runs. In general, using more runs will result in less variable results.

### Storing variables

Online inference can be performed without storing all the past values of all variables inferred by the COIN model. To reduce memory requirements, the past values of variables are only stored if needed for analysis. To store specific variables across trials, add the names of the variables to the store property. For example, to store the Kalman gains and responsibilities:
```
obj.store = {'k','cPost'};
```
This property must be set before running the model. The stored variables can be analysed after running the model. For example, to compute the Kalman gain of the context with the highest responsibility for each particle on each trial:
```
for trial = 1:numel(obj.x) % loop over trials
    for particle = 1:obj.P % loop over particles
        [~,j] = max(S{1}.cPost(:,particle,trial));
        k(particle,trial) = S{1}.k(j,particle,trial);
    end
end
```
This result can be averaged across particles on each trial as all particles within the same run have the same weight. For a full list of the names of variables that can be stored see [Variable names](#variable-names).

### Fitting the model to data

The parameters of the COIN model can be fit to data using maximum likelihood estimation. Use the adaptation property to pass data to the model. The data should be in vector form with one element per trial (use NaN on trials where adaptation is not measured). After the model parameters and paradigm have been defined (see [Properties](#properties)), the negative log likelihood of the data can be estimated by calling the objective_COIN method on object obj:
```
o = obj.objective_COIN;
```
This method returns a stochastic estimate of the objective. It is stochastic because it is calculated from simulations that are conditioned on random observation noise. To aid parameter optimisation, the variance of this estimate can be reduced by increasing the number of runs via the R property (this is best done in conjunction with [Parallel computing](#parallel-computing) to avoid prohibitively long runtimes). The estimate of the objective can be passed to an optimiser. An optimiser that can handle a stochastic objective function should be used (e.g. [BADS](https://github.com/lacerbi/bads)).

### Inferring internal representations of the COIN model fit to adaptation data


### Using adaptation data to assign weights to runs

After fitting the model to data, the internal representations that generated the data can be inferred from the data. This can be done by defining the adaptation property before calling the run_COIN method. This will result in each run being assigned a weight based on how well it explains the adaptation data. In general, these weights will not be equal (although they can be if the runs were recently resampled). When averaging variables or distributions across runs, these weights should be used to compute a weighted average.



The sequence of observation noise that a participant perceives is unknown. However, some sequences are more probable than others based on the adaptation data of a participant.
In a [previous section](#running-the-model), we simulated the COIN model by performing many runs on the same paradigm using the same parameters. Each run was conditioned on a different random sequence of observation noise and was assigned an equal weight. If the parameters used to perform the simulation were fit to data, this data can be
simulation by specifying the model parameters and the paradigm. 

### Parallel computing

It is possible to obtain better fits to data and cleaner internal representations by increasing the number of runs. However, 

The computational complexity of the COIN model scales linearly with the number of runs. To reduce runtime, each run can be performed in parallel across multiple CPU cores (e.g. on a computer cluster). To engage parallel processing, use the maxCores property to specify the maximum number of CPU cores available for use. The default setting of maxCores is 0, which implements serial processing.

This simulation performed inference based on a single sample of the observation noise, which transforms the perturbation into the state feedback. 

### Integrating out observation noise
The basic simulation above have performed inference conditioned on a random sequence of observation noise.
We can repeat the simulation multiple times (each based on a different sequence of observation noise) using the property *R*. For example, to run 2 simulations, call
The output of each and each simulation, or run, is assigned a weight (*w*). In the absence of behavioural adaptation data, all runs are assigned equal weight. If we pass adaptation data via the adaptation property
The more simulations we perform (the greater R is), the greater the computational complexity. If you have access to a computer cluster, you can perform each simulation in parallel, which will speed things up. To do this, specify the maximum number of CPU cores you have access to via the property *maxCores*.

### Properties

### Variable names

note that xpred, vpred and cpred are stored before resampling

