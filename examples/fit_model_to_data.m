% number of participants
P = 8;

% number of trials
trials = 200;

% preallocate memory
perturbations = zeros(P,trials);
cues = zeros(P,trials);
adaptation = zeros(P,trials);

% create synthetic data for each participant
for p = 1:P

    % create an object
    obj = COIN;
    
    % sequence of perturbations
    obj.perturbations = [zeros(1,50) ones(1,100) zeros(1,50)];
    
    % insert a channel trial (randomly) every block of 10 trials
    channel_trials = (1:10:trials) + randi(9,[1,20]);
    obj.perturbations(channel_trials) = NaN; % define channel trials as NaN perturbations
    
    % sequence of sensory cues
    obj.cues = [ones(1,50) 2*ones(1,100) ones(1,50)];
    
    % store the perturbations and sensory cues for use when fitting
    perturbations(p,:) = obj.perturbations;
    cues(p,:) = obj.cues;
    
    % run the COIN model
    S = obj.simulate_COIN;

    % noiseless motor output
    noiseless_motor_output = S.runs{1}.yHat;
    
    % random motor noise
    motor_noise = randn(trials,1)*obj.sigma_motor_noise;
    
    % adaptation (noisy motor output)
    adaptation(p,:) = noiseless_motor_output + motor_noise;
    
    % set adaptation to NaN on trials where adaptation was not measured
    non_channel_trials = setdiff(1:trials,channel_trials);
    adaptation(p,non_channel_trials) = NaN;
    
end

% BADS options
options = [];
options.UncertaintyHandling = 1; % tell BADS that the objective is noisy
options.NoiseFinalSamples = 1; % # samples to estimate FVAL at the end (default would be 10) - use more if obj.runs is small

% create an array of COIN objects (one per participant)
object = create_object_array(perturbations,cues,adaptation);

% define the objective to be optimised (the negative log-likelihood)
objective = @(param) fit_COIN(object,param);

% DON'T TAKE THESE BOUNDS AS GOSPEL! YOU MAY WANT TO NARROW/SHIFT/WIDEN THEM
lb  = [1e-3 0.2    1   1   1e-2 1e-6 1e-4   1e-6]; % lower bounds
plb = [1e-2 0.5    10  10  5e-2 1e-2 1e-3   1e-4]; % plausible lower bounds
pub = [2e-1 0.9    1e5 1e7 2e-1 1e5  0.9    1e2 ]; % plausible upper bounds
ub  = [5e-1 1-1e-6 1e7 1e9 3e-1 1e6  1-1e-4 1e4 ]; % upper bounds

% random initial parameters inside plausible region
x0 = plb + (pub-plb).*rand(1,numel(plb));

% fit the COIN model to the synthetic data using BADS
[maximum_likelihood_parameters,fval,exitflag,output] = bads(objective,x0,lb,ub,plb,pub,[],options);

%% plot the fit

object2 = object;

% simulate the model for each participant using the fitted parameters
for p = 1:P
    
    % set the parameters to their maximum-likelihood values
    set_parameters(object2(p),maximum_likelihood_parameters);

    % run the COIN
    S = object2(p).simulate_COIN;

    % store the noiseless motor output
    adaptation_fitted(p,:) = S.runs{1}.yHat;
    
end

for p = 1:P
    
    data(p,:) = adaptation(p,~isnan(adaptation(p,:)));
    
end

figure
hold on
plot([zeros(1,50) ones(1,100) zeros(1,50)],'k','LineWidth',2)
adaptation_fitted = rand(2,200);
y = mean(data);
err = std(data)/sqrt(P);
errorbar(5:10:200,y,err,'r','LineWidth',2)
plot(mean(adaptation_fitted,1),'b','LineWidth',2)
legend('perturbation','synthetic data','model fit')
ylabel('adaptation')
xlabel('trial')

function obj = create_object_array(perturbations,cues,adaptation)

    % number of participants
    P = size(adaptation,1);

    % array of objects (one object per participant)
    obj(1,P) = COIN;
    
    % define object properties for each participant
    for p = 1:P

        % sequence of perturbations (can be unique to each participant)
        obj(p).perturbations = perturbations(p,:);
        
        % sequence of sensory cues (can be unique to each participant)
        obj(p).cues = cues(p,:);

        % adaptation (unique to each participant)
        obj(p).adaptation = adaptation(p,:);
            
        % number of runs (observation noise sequences)
        obj(p).runs = 10;
            
        % number of CPUs available (0 performs serial processing)
        obj(p).max_cores = 2;

    end
    
end

function negative_log_likelihood = fit_COIN(obj,param)

    % number of participants
    P = length(obj);

    % set the parameters for each participant
    for p = 1:P
        
        set_parameters(obj(p),param);
        
    end
    
    % compute the objective
    negative_log_likelihood = obj.objective_COIN;
    
end
    
function set_parameters(obj,param)  

    % parameters (shared by all participants)
    obj.sigma_process_noise = param(1);         % standard deviation of process noise
    obj.prior_mean_retention = param(2);        % prior mean of retention
    obj.prior_precision_retention = param(3)^2; % prior precision (inverse variance) of retention
    obj.prior_precision_drift = param(4)^2;     % prior precision (inverse variance) of drift
    obj.sigma_motor_noise = param(5);           % standard deviation of motor noise
    obj.alpha_context = param(6);               % alpha hyperparameter of the Chinese restaurant franchise for the context transitions
    obj.rho_context = param(7);                 % rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context transitions
    obj.alpha_cue = param(8);                   % alpha hyperparameter of the Chinese restaurant franchise for the cue emissions
    
end
