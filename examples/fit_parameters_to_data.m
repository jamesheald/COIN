% specify the number of participants here
% if P = 1, the parameters of the COIN model are fit to the data of an individual participant
% if P > 1, the parameters of the COIN model are fit to the average data of a group of P participants
P = 1;

% a pre-exposure (P^0), exposure (P^+ and P^-), post-exposure (P^0) paradigm with two sensory cues
pre_exposure_blocks = 4;
exposure_blocks = 52;
post_exposure_blocks = 12;
number_of_blocks = pre_exposure_blocks + exposure_blocks + post_exposure_blocks;
trials_per_block = 8;
trials = number_of_blocks*trials_per_block;

% preallocate memory
perturbations = zeros(P,trials);
cues = zeros(P,trials);
adaptation = zeros(P,trials);

% create synthetic adaptation data for each participant
for participant = 1:P

    % create an object of the COIN class
    obj = COIN;
    
    % generate a random sequence of two sensory cues
    % each cue should occur an equal number of times in each block of trials
    block_of_cues = [ones(1,trials_per_block/2) 2*ones(1,trials_per_block/2)];
    for i = 1:number_of_blocks
        cues(participant,trials_per_block*(i-1) + (1:trials_per_block)) = datasample(block_of_cues,trials_per_block,'Replace',false);
    end
    obj.cues = cues(participant,:);
    
    % ensure the first cue observed is cue 1 and the second cue observed is cue 2
    obj.renumber_cues;
    cues(participant,:) = obj.cues;

    % generate a sequence of perturbations
    % in the exposure phase, each perturbation (1 vs. -1) is paired with a different sensory cue
    exposure_trials = (pre_exposure_blocks*trials_per_block) + (1:exposure_blocks*trials_per_block);
    perturbations(participant,intersect(exposure_trials,find(cues(participant,:)==1))) = 1;
    perturbations(participant,intersect(exposure_trials,find(cues(participant,:)==2))) = -1;
    obj.perturbations = perturbations(participant,:);
    
    % insert one channel trial in every block of trials
    % the location of the channel trial should be random but such that there is one channel trial per cue for every 2 blocks of trials
    for i = 1:number_of_blocks
        if mod(i,2)
            chosen_cue = datasample([1 2],1);
        else
            chosen_cue = setdiff([1 2],chosen_cue);
        end
        chosen_cue_locations = find(cues(participant,trials_per_block*(i-1) + (1:trials_per_block))==chosen_cue);
        y = datasample(chosen_cue_locations,1);
        channel_trial = trials_per_block*(i-1) + y;
        obj.perturbations(channel_trial) = NaN; % indicate a channel trial by setting the perturbation to NaN
    end

    % run a COIN model simulation
    OUTPUT = obj.simulate_COIN;

    % noiseless motor output
    noiseless_motor_output = OUTPUT.runs{1}.motor_output;
    
    % random motor noise
    motor_noise = randn(trials,1)*obj.sigma_motor_noise;
    
    % adaptation
    adaptation(participant,:) = noiseless_motor_output + motor_noise;
    
    % create an adaptation vector (use NaN on trials where adaptation was not measured)
    channel_trials = find(isnan(obj.perturbations));
    non_channel_trials = setdiff(1:trials,channel_trials);
    adaptation(participant,non_channel_trials) = NaN;
    
end

% number of runs that are used to estimate the negative log-likelihood
runs = 10;
warning('consider increasing the number of runs that are used to estimate the negative log-likelihood')
            
% number of CPUs available (0 performs serial processing)
max_cores = feature('numcores');
warning('consider performing parallel computing on a computer cluster with more CPUs to speed up parameter optimisation')

% create an array of objects of the COIN class (one object in the array per participant)
object_fit = create_object_array(perturbations,cues,adaptation,runs,max_cores);

% define the objective to be optimised (the negative log-likelihood)
objective = @(param) fit_COIN(object_fit,param);

warning('consider changing the parameter bounds used by BADS')
lb  = [1e-3 0.2    1   1   1e-2 1e-6 1e-4   1e-6]; % lower bounds
plb = [1e-2 0.5    10  10  5e-2 1e-2 1e-3   1e-4]; % plausible lower bounds
pub = [2e-1 0.9    1e5 1e7 2e-1 1e5  0.9    1e2 ]; % plausible upper bounds
ub  = [5e-1 1-1e-6 1e7 1e9 3e-1 1e6  1-1e-4 1e4 ]; % upper bounds

% random initial parameters inside plausible region
x0 = plb + (pub-plb).*rand(1,numel(plb));

% BADS options
options = [];
options.UncertaintyHandling = 1; % tell BADS that the objective is noisy
options.NoiseFinalSamples = 1; % # samples to estimate FVAL at the end (default is 10) - considering increasing if obj.runs is small

% fit the COIN model to the average synthetic adaptation data using BADS
[maximum_likelihood_parameters,fval,exitflag,output] = bads(objective,x0,lb,ub,plb,pub,[],options);
warning('consider repeating parameter optimisation from multiple initial points')

%% simulate the COIN model with the fitted parameters and plot the fit

% number of runs used to simulate the model with the fitted parameters
% this does not need to be the same as the number of runs used to estimate the negative log-likelihood
runs = 10;
warning('consider increasing the number of runs of the COIN model simulation')

% create an array of objects
object_simulate = create_object_array(perturbations,cues,adaptation,runs,max_cores);

% simulate the model for each participant using the fitted parameters
data = zeros(P,number_of_blocks/2,2);
fit = zeros(P,number_of_blocks/2,2);
for participant = 1:P
    
    % set the parameters to their fitted (maximum-likelihood) values
    set_parameters(object_simulate(participant),maximum_likelihood_parameters);
    
    if P == 1
        object_simulate(participant).adaptation = adaptation(participant,:);
    elseif P > 1
        object_simulate(participant).adaptation = [];
    end
    
    % plot internal representations
    if P == 1
        object_simulate(participant).plot_predicted_probabilities = true;
        object_simulate(participant).plot_state_given_context = true;
        object_simulate(participant).plot_state = true;
        object_simulate(participant).plot_local_transition_probabilities = true;
        object_simulate(participant).plot_global_transition_probabilities = true;
        object_simulate(participant).plot_local_cue_probabilities = true;
        object_simulate(participant).plot_global_cue_probabilities = true;
    end

    % run the COIN model simulation
    OUTPUT = object_simulate(participant).simulate_COIN;
    
    % extract the motor output of each run of the simulation
    motor_output = zeros(trials,runs);
    for run = 1:runs
        motor_output(:,run) = OUTPUT.runs{run}.motor_output;
    end
    
    % find the channel trials for each cue
    cue1_channel_trials = cues(participant,:) == 1 & ~isnan(adaptation(participant,:));
    cue2_channel_trials = cues(participant,:) == 2 & ~isnan(adaptation(participant,:));
    
    % store the adaptation data separately for each perturbation sign
    if any(perturbations(participant,cue1_channel_trials) == 1)
        data(participant,:,1) = adaptation(participant,cue1_channel_trials);
        data(participant,:,2) = adaptation(participant,cue2_channel_trials);
        fit(participant,:,1) = sum(OUTPUT.weights.*motor_output(cue1_channel_trials,:),2);
        fit(participant,:,2) = sum(OUTPUT.weights.*motor_output(cue2_channel_trials,:),2);
    elseif any(perturbations(participant,cue1_channel_trials) == -1)
        data(participant,:,1) = adaptation(participant,cue2_channel_trials);
        data(participant,:,2) = adaptation(participant,cue1_channel_trials);
        fit(participant,:,1) = sum(OUTPUT.weights.*motor_output(cue2_channel_trials,:),2);
        fit(participant,:,2) = sum(OUTPUT.weights.*motor_output(cue1_channel_trials,:),2);
    end
    
end

figure
hold on
y = mean(data(:,:,1),1);
err = std(data(:,:,1),[],1)/sqrt(P);
h = errorbar(1:trials_per_block*2:trials,y,err,'r','LineWidth',1);
alpha = 0.3;
set([h.Bar, h.Line], 'ColorType', 'truecoloralpha', 'ColorData', [h.Line.ColorData(1:3); 255*alpha])
set(h.Cap, 'EdgeColorType', 'truecoloralpha', 'EdgeColorData', [h.Cap.EdgeColorData(1:3); 255*alpha])
plot(1:trials_per_block*2:trials,mean(fit(:,:,1),1),'r','LineWidth',2)
y = mean(data(:,:,2),1);
err = std(data(:,:,2),[],1)/sqrt(P);
h = errorbar(1:trials_per_block*2:trials,y,err,'b','LineWidth',1);
set([h.Bar, h.Line], 'ColorType', 'truecoloralpha', 'ColorData', [h.Line.ColorData(1:3); 255*alpha])
set(h.Cap, 'EdgeColorType', 'truecoloralpha', 'EdgeColorData', [h.Cap.EdgeColorData(1:3); 255*alpha])
plot(1:trials_per_block*2:trials,mean(fit(:,:,2),1),'b','LineWidth',2)
l = legend('data (P^+ perturbation)','fit (P^+ perturbation)','data (P^- perturbation)','fit (P^- perturbation)','box','off','AutoUpdate','off');
l.Position(1:2) = [0.5 0.57];
plot([0 trials],[0 0],'k--')
axis([0 trials -1 1])
set(gca,'YTick',[-1 0 1],'XTick',[0 trials],'FontSize',10)
ylabel('adaptation')
xlabel('trial')

function obj = create_object_array(perturbations,cues,adaptation,runs,max_cores)

    % number of participants
    P = size(adaptation,1);

    % array of objects (one object per participant)
    obj(1,P) = COIN;
    
    % define object properties for each participant
    for participant = 1:P

        % sequence of perturbations (may be unique to each participant)
        obj(participant).perturbations = perturbations(participant,:);
        
        % sequence of sensory cues (may be unique to each participant)
        obj(participant).cues = cues(participant,:);

        % adaptation (unique to each participant)
        obj(participant).adaptation = adaptation(participant,:);
            
        % number of runs
        obj(participant).runs = runs;
            
        % number of CPUs available (0 performs serial processing)
        obj(participant).max_cores = max_cores;

    end
    
end

function negative_log_likelihood = fit_COIN(obj,param)

    % number of participants
    P = length(obj);

    % set the parameters for each participant
    for participant = 1:P
        
        set_parameters(obj(participant),param);
        
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
