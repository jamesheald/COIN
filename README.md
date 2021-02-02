# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. The model is described in detail in a bioRxiv preprint: [Contextual inference underlies the learning of sensorimotor repertoires](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1).

## Dependencies

To improve the efficiency of the code, the COIN model requires installation of the following packages:

- "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. Run install_lightspeed.m to compile the necessary mex files. 
- "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh. Run make.m to compile the necessary mex files. 
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
This object has [properties](#properties) that define the model (e.g., model parameters) and the paradigm (e.g., perturbations, sensory cues). Some properties have default values, which can be overwritten. The default model parameters were fit to the average spontaneous and evoked recovery data sets in [Contextual inference underlies the learning of sensorimotor repertoires](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1). Data in properties can be operated on by methods (functions) of the class.

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

The actual observation noise that a participant perceives is unknown. Therefore, rather than performing a single run of a simulation conditioned on one particular noise sequence, multiple runs of the simulation can be performed, each conditioned on a different noise sequence. The observation noise can then be integrated out by computing the average value of each variable inferred by the COIN model across runs. To specify the number of runs to perform, use the R property. For example, to perform 2 runs:
```
obj.R = 2;
[S,w] = obj.run_COIN;
```
S is a cell array with one cell per run and w is vector specifying the relative weight of each run. In this example, each run is assigned an equal weight. However, if adaptation data is passed to the object before running the model, each run will be assigned a weight based on how well it explains the data (see [Inferring internal representations fit to adaptation data](#inferring-internal-representations-of-the-COIN-model-fit-to-adaptation-data)).

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
P = obj.plot_COIN(S,w);
```
This will generate the desired plots&mdash;a state | context plot and a predicted probabilities plot in this example. The structure P contains the data that is plotted (view the generate_figures method in COIN.m to see how the data in P is plotted). Note that these plots may take some time to generate, as they require contexts in multiple particles and multiple runs to be relabelled on each trial. Once the contexts have been relabelled, the relevant variables or distributions are averaged across particles and runs. In general, using more runs will result in less variable results.

### Storing variables

Online inference can be performed without storing all of the past values of all inferred variables. Therefore, to reduce memory requirements, the past values of variables are only stored if they will be needed for later analysis. The only exceptions to this are the adaptation and the state feedback, which are always stored. To store the values of particular variables on all trials, add the names of these variables to the store property. For example, to store the Kalman gains and responsibilities:
```
obj.store = {'k','cFilt'};
```
This property must be set before running the model. The stored variables can be analysed after the model has been run. For example, to compute the Kalman gain of the context with the highest responsibility:
```
for trial = 1:numel(obj.x) % loop over trials
    for particle = 1:obj.P % loop over particles
        [~,j] = max(S{1}.cFilt(:,particle,trial));
        k(particle,trial) = S{1}.k(j,particle,trial);
    end
end
```
A simple average across particles can be computed on each trial, as all particles within the same run have the same weight. For a full list of the names of variables that can be stored see [Variable names](#variable-names).

### Fitting the model to data

#### Individual participants’ data

The COIN model is fit to data by finding the parameters that minimise the negative log of the likelihood function. To calculate the negative log-likelihood, define the model parameters (see [Properties](#properties)) and pass the data to the class object via the adaptation property. The data should be in vector form with one element per trial (use NaN on trials where adaptation was not measured). The objective_COIN method can then be called:
```
o = obj.objective_COIN;
```
This returns a stochastic estimate of the negative log-likelihood, which can be passed to an optimiser. The estimate is stochastic because it is calculated from simulations that are conditioned on random observation noise. To aid parameter optimisation, the variance of the estimate can be reduced by increasing the number of runs using the R property (this is best done in conjunction with [Parallel computing](#parallel-computing) to avoid long runtimes). An optimiser that can handle a stochastic objective function should also be used (e.g., [BADS](https://github.com/lacerbi/bads)).

The COIN model is fit to data by finding the parameters that minimise the negative log of the likelihood function. 

#### Group average data

To calculate the negative log-likelihood for group average data, create an array of objects (one per participant):
```
for p = 1:P % loop over participants
    
    % object for participant p
    obj(p) = COIN;
    
    % parameters (same for all participants)
    obj(p).sigmaQ = 0.0089;                          % standard deviation of process noise
    obj(p).adMu = [0.9425 0]                         % mean of prior of retention and drift
    obj(p).adLambda = diag([837.1539 1.2227e+03].^2) % precision of prior of retention and drift
    obj(p).sigmaM = 0.0182                           % standard deviation of motor noise
    obj(p).alpha = 8.9556                            % alpha hyperparameter of the Chinese restaurant franchise for the context
    obj(p).rho = 0.2501                              % rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context
    
    % paradigm (may be unique to each participant)
    obj(p).x = ...;
    
    % adaptation (unique to each participant)
    obj(p).adaptation = ...;
    
end
```
As shown above, use ([properties](#properties)) to define the model parameters, paradigm and adaptation data. The adaptation data should be in vector form with one element per trial (use NaN on trials where adaptation was not measured). There are assumed to be an equal number of adaptation data points per participant (the *i*-th average adaptation data point is the average *i*-th adaptation data point across participants). The objective_COIN method can then be called:
```
o = obj.objective_COIN;
```
This returns a stochastic estimate of the negative log-likelihood, which can be passed to an optimiser. The estimate is stochastic because it is calculated from simulations that are conditioned on random observation noise. To aid parameter optimisation, the variance of the estimate can be reduced by increasing the number of runs using the R property (this is best done in conjunction with [Parallel computing](#parallel-computing) to avoid long runtimes). An optimiser that can handle a stochastic objective function should also be used (e.g., [BADS](https://github.com/lacerbi/bads)).

#### Individual participants’ data

To calculate the negative log-likelihood for an individual participants’ data, the proceedure is the same (an individual participant is a special case of a group with 1 member).

 and then call the objective_COIN method:
 For each object in the array, define the model parameters (see [Properties](#properties)) and pass the data to the class object via the adaptation property. 

### Inferring internal representations of the COIN model fit to adaptation data

When multiple runs are performed using parameters fit to data, each run can be assigned a weight based on how well it explains the data. In general, these weights will not be equal (although they can be, as weights are reset when runs are resampled during particle filtering). To generate a set of weighted runs, set the model parameters to their maximum likelihood estimates, pass the data to the class object via the adaptation property and call the run_COIN method. The resultant weights should be used to compute a weighted average of variables across runs.

### Parallel computing

The time it takes to execute multiple runs in series (using a for loop) can be prohibitively long if the number of runs is large. To execute multiple runs in parallel (e.g., across different CPU cores on a computer cluster), specify the maximum number of CPU cores available using the maxCores property. The default setting of maxCores is 0, which executes runs in series.

### Properties
Below is a list of all the properties of the COIN class. A brief description of each property is provided.
```
% model implementation
P = 100                                   % number of particles
R = 1                                     % number of runs to perform, each conditioned on a different observation noise sequence
maxC = 10                                 % maximum number of contexts that the model can instantiate
maxCores = 0                              % maximum number of CPU cores available (0 implements serial computing)

% model parameters
sigmaQ = 0.0089                           % standard deviation of process noise
adMu = [0.9425 0]                         % mean of prior of retention and drift
adLambda = diag([837.1539 1.2227e+03].^2) % precision of prior of retention and drift
sigmaS = 0.03                             % standard deviation of sensory noise
sigmaM = 0.0182                           % standard deviation of motor noise
learnB = false                            % learn the measurment bias or not
bMu = 0                                   % mean of prior of measurement bias (if bias is being learned)
bLambda = 70^2                            % precision of prior of measurement bias (if bias is being learned)
gamma = 0.1                               % gamma hyperparameter of the Chinese restaurant franchise for the context
alpha = 8.9556                            % alpha hyperparameter of the Chinese restaurant franchise for the context
rho = 0.2501                              % rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context
gammaE = 0.1                              % gamma hyperparameter of the Chinese restaurant franchise for the cue
alphaE = 0.1                              % alpha hyperparameter of the Chinese restaurant franchise for the cue
sigmaR                                    % standard deviation of observation noise (set later based on sigmaS and sigmaM)
kappa                                     % kappa (self-transition) hyperparameter of the Chinese restaurant franchise for the context (set later based on rho and alpha)
H                                         % matrix of context-dependent observation vectors (set later)

% paradigm
x                                         % vector of (noiseless) perturbations (NaN if channel trial)
q                                         % vector of sensory cues (if any)
cuesExist                                 % does the experiment have sensory cues or not (set later based on q)
T                                         % total number of trials (set later based on the length of x)
trials                                    % trials to simulate (set later to 1:T if empty)
observeY                                  % is the state feedback observed or not on each trial (set later based on x)
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
xHatPlot                                  % plot mean of state (marginal distribution). if learnB == false, xHat = yHat

% plot inputs
gridA                                     % if adPlot == true, specify values of a to evaluate p(a) at
gridD                                     % if adPlot == true, specify values of d to evaluate p(d) at
gridX                                     % if xPredPlot == ture or xPredMargPlot == true, specify values of x to evaluate p(x) at
gridB                                     % if bPlot == true or bPredMargPlot == true, specify values of b to evaluate p(b) at
gridY                                     % if yPredMargPlot == true, specify values of y to evaluate p(y) at
```

### Variable names
Below is a list of all the COIN model variables that can be stored. A brief description of each variable is provided. Note that variables in bold are stored before particles are resampled to ensure they do not depend on the state feedback. Care should be taken if these variables are analysed in conjunction with variables that are stored after particles are resampled (the indices of resampled particles need to be considered).
<pre>
a                % retention in each context
adCovar          % covariance of the posterior of the retention and drift in each context
adMu             % mean of the posterior of the retention and drift in each context
adSS1            % sufficient statistic for the retention and drift parameters #1
adSS2            % sufficient statistic for the retention and drift parameters #2
b                % bias in each context
beta             % global transition distribution
betaE            % global emission distribution  
bMu              % mean of the posterior of the bias in each context
bPredMarg        % bias (marginal distribution, discretised)
bSS1             % sufficient statistic for the bias parameter #1
bSS2             % sufficient statistic for the bias parameter #2
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
p                % variance of state feedback predicted in each context
pQ               % probability of the observed cue in each context        
pY               % probability of the observed state feedback in each context
Q                % number of cues observed so far
sensoryNoise     % sensory noise
transitionMatrix % expected value of the context transition matrix
vFilt            % variance of the filtered distribution of the state in each context 
<b>vPred            % variance of the predictive distribution of the state in each context</b> 
vPrev            % variance of the filtered distribution of the state in each context 
xFilt            % mean of the filtered distribution of the state in each context 
xHat             % state predicted (average across contexts and particles)
<b>xPred            % mean of the predictive distribution of the state in each context</b>
xPredMarg        % predicted state (marginal distribution, discretised)
xPrev            % mean of the filtered distribution of the state in each context on previous trial
xPrevSamp        % state sampled on the previous trial (used to update the sufficient statistics for the retention and drift parameters)
xSamp            % state sampled on the current trial (used to update the sufficient statistics for the bias parameter)
xSampOld         % state sampled on the current trial (used to update the sufficient statistics for the retention and drift parameters)
yHat             % state feedback predicted (average across contexts and particles)
yPred            % state feedback predicted in each context
yPredMarg        % predicted state feedback (marginal distribution, discretised)
</pre>
