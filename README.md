# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. The model is described in detail in a [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1) paper.

## Dependencies

The COIN model requires installation of the following packages to improve the efficiency of the code:

- "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. To compile the necessary mex files, run install_lightspeed.m.
- "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh. To compile the necessary mex files, run make.m.
- "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each package to the MATLAB search path: 
```
addpath(genpath('directory'))
```
Here 'directory' is the full path to the root folder of the package.

## The model

### Running the model

The COIN model is implemented as a class in MATLAB. An object of the class can be created by calling
```
obj = COIN;
```
This object has [properties](#properties) that define the model (e.g., model parameters) and the paradigm (e.g., perturbations, sensory cues). Some properties have default values, which can be overwritten. Data in these properties can be operated on by methods (functions) of the class.

To simulate learning on a simple paradigm, define a series of perturbations (use NaN to indicate a channel trial):
```
obj.x = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)]; % spontaneous recovery paradigm
```
and run the model by calling the run_COIN method on object obj:
```
[S,w] = obj.run_COIN;
```
The adaptation and state feedback (perturbation plus random observation noise) are contained in a cell in S. To plot them:
```
figure
plot(S{1}.y,'m.')
hold on
plot(S{1}.yHat,'c')
legend('state feedback','adaptation')
```
### Integrating out observation noise

The actual observation noise that a participant perceives is unknown. Therefore, rather than performing a single run of a simulation conditioned on one particular noise sequence, multiple runs of the simulation can be performed, each conditioned on a different noise sequence. Inferences can be averaged across runs, thus integrating out the observation noise. To specify the number of runs to perform, use the R property. For example, to perform 2 runs:
```
obj.R = 2;
[S,w] = obj.run_COIN;
```
S is a cell array with one cell per run and w is vector specifying the relative weight of each run. In this example, each run is assigned an equal weight. However, if adaptation data is passed to the class object, each run will be assigned a weight based on how well it explains the data (see [Inferring internal representations fit to adaptation data](#inferring-internal-representations-fit-to-adaptation-data)).

### Plotting internal representations

To plot specific variables or distributions, activate the relevant plot flags in the properties of the class object. If plotting a continuous distribution, also specify the points at which to evaluate the distribution. For example, to plot the distribution of the state of each context and the predicted probabilities:
```
obj.xPredPlot = true; % i want to plot the state | context
obj.gridX = linspace(-1.5,1.5,500); % points to evaluate state | context at
obj.cPredPlot = true; % i want to plot the predicted probabilities
```
These properties must be set before running the model so that the necessary variables can be stored. For a list of variables that can be plotted and how to plot them, see [Properties](#properties).

After the model has been run, call the plot_COIN method:
```
[P,S] = obj.plot_COIN(S,w);
```
This will generate the desired plots&mdash;a state | context plot and a predicted probabilities plot in this example. The structure P contains the data that is plotted (view the generate_figures method in COIN.m to see how the data in P is plotted). Note that these plots may take some time to generate, as they require contexts in multiple particles and multiple runs to be relabelled on each trial. Once the contexts have been relabelled, the relevant variables or distributions are averaged across particles and runs. In general, using more runs will result in less variable results.

### Storing variables

Online inference can be performed without storing all the past values of all variables inferred by the COIN model. Hence, to reduce memory requirements, the past values of variables are only stored if needed for later analysis. To store the values of particular variables on all trials, add the names of these variables to the store property. For example, to store the Kalman gains and responsibilities:
```
obj.store = {'k','cPost'};
```
This property must be set before running the model. The stored variables can be analysed after the model has been run. For example, to compute the Kalman gain of the context with the highest responsibility for each particle on each trial:
```
for trial = 1:numel(obj.x) % loop over trials
    for particle = 1:obj.P % loop over particles
        [~,j] = max(S{1}.cPost(:,particle,trial));
        k(particle,trial) = S{1}.k(j,particle,trial);
    end
end
```
This result can be averaged across particles on each trial, as all particles within the same run have the same weight. For a full list of the names of variables that can be stored see [Variable names](#variable-names).

### Fitting the model to data

The COIN model is fit to data by finding the parameters that minimise the negative log of the likelihood function. To calculate the negative log-likelihood, define the model parameters (see [Properties](#properties)) and pass the data to the class object via the adaptation property. The data should be in vector form with one element per trial (use NaN on trials where adaptation was not measured). The objective_COIN method can then be called:
```
o = obj.objective_COIN;
```
This returns a stochastic estimate of the negative log-likelihood, which can be passed to an optimiser. The estimate is stochastic because it is calculated from simulations that are conditioned on random observation noise. To aid parameter optimisation, the variance of the estimate can be reduced by increasing the number of runs via the R property (this is best done in conjunction with [Parallel computing](#parallel-computing) to avoid long runtimes). An optimiser that can handle a stochastic objective function should also be used (e.g., [BADS](https://github.com/lacerbi/bads)).

### Inferring internal representations of the COIN model fit to adaptation data

When multiple runs of a simulation are performed using parameters fit to data, each run of the simulation can be assigned a weight based on how well it explains the data. In general, these weights will not be equal (although they can be, as weights are reset when runs are resampled during particle filtering). To generate a set of weighted runs, set the model parameters to their maximum likelihood estimates, pass the data to the class object via the adaptation property and call the run_COIN method. The resultant weights should be used when averaging inferences across runs.

### Parallel computing

The time it takes to execute multiple runs of a simulation in series (using a for loop) can be prohibitively long if the number of runs is large. To execute multiple runs in parallel (e.g., across different CPU cores on a computer cluster), specify the maximum number of CPU cores available using the maxCores property. The default setting of maxCores is 0, which executes multiple runs in series.

### Properties

### Variable names

note that xpred, vpred and cpred are stored before resampling

