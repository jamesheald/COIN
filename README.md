# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. The model is described in detail in a recent bioRxiv preprint: [Contextual inference underlies the learning of sensorimotor repertoires](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1).

## Installation

1. Download the COIN.m file.
2. Install the following packages (to improve the efficiency of the code):
   - "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. 
     - Run install_lightspeed.m to compile the necessary mex functions. 
   - "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh.
     - Run make.m to compile the necessary mex functions. 
   - "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each package to the MATLAB search path using the following command: 
```
addpath(genpath('directory'));
```
Here 'directory' is the full path to the root folder of the package.

## The model

### Running the model

The COIN model is implemented as a class in MATLAB. An object of the class can be created by calling
```
obj = COIN;
```
This object has [properties](#properties) that define the model (e.g., model parameters) and the paradigm (e.g., perturbations, sensory cues). Some properties have default values, which can be overwritten. Data in properties can be operated on by methods (functions) of the class.

To simulate learning on a simple paradigm, first define a series of perturbations (use NaN to indicate a channel trial):
```
obj.x = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)]; % spontaneous recovery paradigm
```
Then run the model by calling the run_COIN method on object obj:
```
[S,w] = obj.run_COIN;
```
The adaptation and state feedback (perturbation plus random observation noise) are contained in a cell in S. They can be plotted as follows:
```
figure
plot(S{1}.y,'.')
hold on
plot(S{1}.yHat)
legend('state feedback','adaptation')
```
### Integrating out observation noise

The actual observation noise that a participant perceives is unknown. Therefore, rather than performing a single run of a simulation conditioned on one particular noise sequence, multiple runs of the simulation can be performed, each conditioned on a different noise sequence. The observation noise can then be integrated out by computing the average value of each variable inferred by the COIN model across runs. To specify the number of runs to perform, use the R property. For example, to perform 2 runs:
```
obj.R = 2;
[S,w] = obj.run_COIN;
```
S is a cell array with one cell per run and w is vector specifying the relative weight of each run. In this example, each run is assigned an equal weight. However, if adaptation data is passed to the object before running the model, each run will be assigned a weight based on how well it explains the data (see [Inferring internal representations fit to adaptation data](#inferring-internal-representations-of-the-COIN-model-fit-to-adaptation-data)).

### Storing variables

Online inference does not require all of the past values of all inferred variables to be stored in memory. Therefore, to reduce memory requirements, the past values of variables are only stored in memory if they will be needed for later analysis (as an exception, the adaptation and state feedback variables are always stored). To store the values of particular variables on all trials, add the names of these variables to the store property. For example, to store the Kalman gains and responsibilities, define store as
```
obj.store = {'k','cFilt'};
```
The store property must be set before calling the run_COIN method. The stored variables can be analysed after the model has been run. As an example analysis, to compute the Kalman gain of the context with the highest responsibility:
```
for run = 1:obj.R % loop over runs
    for trial = 1:numel(obj.x) % loop over trials
        for particle = 1:obj.P % loop over particles
            [~,j] = max(S{run}.cFilt(:,particle,trial));
            k(particle,trial,run) = S{run}.k(j,particle,trial);
        end
    end
end
```
A simple average across particles within a run can be computed on each trial, as all particles within the same run have the same weight. In contrast, a weighted average over runs should be computed on each trial. For a full list of the names of variables that can be stored see [Variable names](#variable-names).

### Plotting internal representations

To plot specific variables or distributions of variables, activate the relevant plot flags in the properties of the class object. If plotting a continuous distribution, also specify the points at which to evaluate the distribution. For example, to plot the distribution of the state of each context and the predicted probabilities:
```
obj.xPredPlot = true; % plot state | context
obj.gridX = linspace(-1.5,1.5,500); % points to evaluate state | context at
obj.cPredPlot = true; % plot the predicted probabilities
```
These properties should be set before calling the run_COIN method so that the variables needed to generate the plots can be stored. For a full list of variables that can be plotted and how to plot them, see the plot flags and plot inputs sections of [Properties](#properties).

After the model has been run, call the plot_COIN method:
```
P = obj.plot_COIN(S,w);
```
This will generate the requested plots&mdash;a state | context plot and a predicted probabilities plot in this example. The structure P contains the data that is plotted (to see how the data in P is plotted, view the generate_figures method). Note that these plots may take some time to generate, as they require contexts in multiple particles and multiple runs to be relabelled on each trial. Once the contexts have been relabelled, the relevant variables or distributions are averaged across particles and runs. In general, using more runs will result in less variable/noisy results.

### Fitting the model to data

The COIN model can be fit either to an individual participants’ data or to the average data of a group of participants. Here, the case of the average group data is presented, as an individual participant is a special case of a group with 1 member.

The COIN model is fit to data by finding the parameters that minimise the negative log of the likelihood function. To calculate the negative log-likelihood, create an object array with one object per participant. For each object, define the relevant model parameters, the paradigm and the adaptation data using the corresponding [properties](#properties) of the object. As an example of how to create an object array (property values still need to be assigned):
```
for p = 1:P % loop over participants
    
    % object for participant p
    obj(p) = COIN;
    
    % parameters (same for all participants)
    obj(p).sigmaQ = ...;              % standard deviation of process noise
    obj(p).adMu = [... 0]             % mean of prior of retention and drift
    obj(p).adLambda = diag([... ...]) % precision of prior of retention and drift
    obj(p).sigmaM = ...               % standard deviation of motor noise
    obj(p).alpha = ...                % alpha hyperparameter of the Chinese restaurant franchise for the context
    obj(p).rho = ...                  % rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context
    
    % paradigm (could be unique to each participant)
    obj(p).x = ...
    
    % adaptation (unique to each participant)
    obj(p).adaptation = ...
    
end
```
The adaptation property should be a vector with one element per trial (use NaN on trials where adaptation was not measured), and the number of adaptation measurements should be the same for all participants (the *n*th average adaptation measurement is the average *n*th adaptation measurement across participants).  

After the object array has been created, call the objective_COIN method:
```
o = obj.objective_COIN;
```
This returns a stochastic estimate of the negative log-likelihood, which can be passed to an optimiser. The estimate is stochastic because it is calculated from simulations that are conditioned on random observation noise. To aid parameter optimisation, the variance of the estimate can be reduced by increasing the number of runs using the R property (this is best done in conjunction with [parallel computing](#parallel-computing) to avoid long runtimes). An optimiser that can handle a stochastic objective function should also be used (e.g., [BADS](https://github.com/lacerbi/bads)).

### Inferring internal representations of the COIN model fit to adaptation data

When multiple runs of a simulation are performed using parameters fit to data, each run can be assigned a weight based on how well it explains the data. In general, these weights will not be equal (although they can be, as weights are reset when runs are resampled during particle filtering). To generate a set of weighted runs, set the model parameters to their maximum likelihood estimates, pass the data to the class object via the adaptation property and call the run_COIN method. The resultant weights should be used to compute a weighted average of variables across runs.

### Parallel computing

The time it takes to execute multiple runs in series (using a for loop) can be prohibitively long if the number of runs is large. To execute multiple runs in parallel (e.g., across different CPU cores on a computer cluster) and thus reduce runtime, specify the maximum number of CPU cores that are available using the maxCores property. The default setting of maxCores is 0, which executes multiple runs in series. Parallel computing is available when calling the run_COIN method (running the model) and the objective_COIN method (fitting the model).

### Properties
Below is a list of all the properties of the COIN class. A brief description of each property is provided.
```
% model implementation
P = 100                                   % number of particles
R = 1                                     % number of runs to perform, each conditioned on a different observation noise sequence
maxC = 10                                 % maximum number of contexts that the model can instantiate
maxCores = 0                              % maximum number of CPU cores available (0 implements serial processing)

% model parameters
sigmaQ = 0.0089                           % standard deviation of process noise
adMu = [0.9425 0]                         % mean of prior of retention and drift
adLambda = diag([837.1539 1.2227e+03].^2) % precision of prior of retention and drift
sigmaS = 0.03                             % standard deviation of sensory noise
sigmaM = 0.0182                           % standard deviation of motor noise
learnB = false                            % learn the measurment bias or not?
bMu = 0                                   % mean of prior of measurement bias (if bias is being learned)
bLambda = 70^2                            % precision of prior of measurement bias (if bias is being learned)
gamma = 0.1                               % gamma hyperparameter of the Chinese restaurant franchise for the context transitions
alpha = 8.9556                            % alpha hyperparameter of the Chinese restaurant franchise for the context transitions
rho = 0.2501                              % rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context transitions
gammaE = 0.1                              % gamma hyperparameter of the Chinese restaurant franchise for the cue emissions
alphaE = 0.1                              % alpha hyperparameter of the Chinese restaurant franchise for the cue emissions
sigmaR                                    % standard deviation of observation noise (set later based on sigmaS and sigmaM)
kappa                                     % kappa (self-transition) hyperparameter of the Chinese restaurant franchise for the context transitions (set later based on rho and alpha)
H                                         % matrix of context-dependent observation vectors (set later)

% paradigm
x                                         % vector of (noiseless) perturbations (NaN if channel trial)
q                                         % vector of sensory cues (use consecutive integers starting from 1 to represent cues)
cuesExist                                 % does the experiment have sensory cues or not (set later based on q)?
T                                         % total number of trials (set later based on the length of x)
trials                                    % trials to simulate (set to 1:T later if empty)
observeY                                  % is the state feedback observed or not on each trial (set later based on x)?
eraserTrials                              % trials on which to overwrite context probabilities with stationary probabilities

% measured adaptation data
adaptation                                % vector of adaptation data (NaN if adaptation not measured on a trial)

% store
store                                     % variables to store in memory

% plot flags
xPredPlot                                 % plot state | context
cPredPlot                                 % plot predicted probabilities
cFiltPlot                                 % plot responsibilities (including novel context probabilities)
cInfPlot                                  % plot stationary probabilities
adPlot                                    % plot retention | context and drift | context
bPlot                                     % plot bias | context
transitionMatrixPlot                      % plot context transition probabilities
emissionMatrixPlot                        % plot cue emission probabilities
xPredMargPlot                             % plot state (marginal distribution)
bPredMargPlot                             % plot bias (marginal distribution)
yPredMargPlot                             % plot predicted state feedback (marginal distribution)
explicitPlot                              % plot explicit component of learning
implicitPlot                              % plot implicit component of learning
xHatPlot                                  % plot mean of state (marginal distribution)

% plot inputs
gridA                                     % if adPlot == true, specify values of a to evaluate p(a) at
gridD                                     % if adPlot == true, specify values of d to evaluate p(d) at
gridX                                     % if xPredPlot == true or xPredMargPlot == true, specify values of x to evaluate p(x) at
gridB                                     % if bPlot == true or bPredMargPlot == true, specify values of b to evaluate p(b) at
gridY                                     % if yPredMargPlot == true, specify values of y to evaluate p(y) at
```

### Variable names
Below is a list of all the COIN model variables that can be stored. A brief description of each variable is provided. Note that variables in bold are stored before particles are resampled so that they do not depend on the state feedback. Care should be taken if these variables are analysed in conjunction with other variables that are stored after particles are resampled (the indices of resampled particles may need to be taken into account).
<pre>
a                % retention in each context
adCovar          % covariance of the posterior of the retention and drift in each context
adMu             % mean of the posterior of the retention and drift in each context
adSS1            % sufficient statistic #1 for the retention and drift parameters in each context
adSS2            % sufficient statistic #2 for the retention and drift parameters in each context
b                % bias in each context
beta             % global transition distribution
betaE            % global emission distribution  
bMu              % mean of the posterior of the bias in each context
bPredMarg        % bias (marginal distribution, discretised)
bSS1             % sufficient statistic #1 for the bias parameter in each context
bSS2             % sufficient statistic #2 for the bias parameter in each context
bVar             % variance of the posterior of the bias in each context
C                % number of instantiated contexts
c                % sampled context
cFilt            % context responsibilities
cInf             % stationary context probabilities
<b>cPred            % predicted context probabilities (conditioned on the cue)</b>
cPrev            % context sampled on the previous trial
cPrior           % prior context probabilities (not conditioned on the cue)
d                % drift in each context
e                % state feedback prediction error in each context
emissionMatrix   % expected value of the cue emission matrix
explicit         % explicit component of learning
implicit         % implicit component of learning
iResamp          % indices of resampled particles
iX               % index of the observed state
k                % Kalman gain in each context
m                % number of tables in restaurant i serving dish j (Chinese restaurant franchise for the context transitions)
mE               % number of tables in restaurant i serving dish j (Chinese restaurant franchise for the cue emissions)
motorNoise       % motor noise
n                % context transition counts
nE               % cue emission counts
<b>pPred            % variance of the predictive distribution of the state feedback in each context</b>
pQ               % probability of the observed cue in each context        
pY               % probability of the observed state feedback in each context
Q                % number of cues observed
sensoryNoise     % sensory noise
transitionMatrix % expected value of the context transition matrix
vFilt            % variance of the filtered distribution of the state in each context 
<b>vPred            % variance of the predictive distribution of the state in each context</b> 
vPrev            % variance of the filtered distribution of the state in each context on previous trial
xFilt            % mean of the filtered distribution of the state in each context 
xHat             % average predicted state (average across contexts and particles)
<b>xPred            % mean of the predictive distribution of the state in each context</b>
xPredMarg        % predicted state (marginal distribution, discretised)
xPrev            % mean of the filtered distribution of the state in each context on previous trial
xPrevSamp        % states sampled on previous trial (to update the sufficient statistics for the retention and drift parameters in each context)
xSamp            % state sampled on current trial (to update the sufficient statistics for the bias parameter in each context)
xSampOld         % states sampled on current trial (to update the sufficient statistics for the retention and drift parameters in each context)
yHat             % average predicted state feedback (average across contexts and particles)
<b>yPred            % mean of the predictive distribution of the state feedback in each context</b>
yPredMarg        % predicted state feedback (marginal distribution, discretised)
</pre>
