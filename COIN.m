classdef COIN < matlab.mixin.Copyable
    % PROPERTIES                                   description
    %     core parameters 
    %         sigma_process_noise                  standard deviation of process noise
    %         sigma_sensory_noise                  standard deviation of sensory noise
    %         sigma_motor_noise                    standard deviation of motor noise
    %         prior_mean_retention                 prior mean of retention
    %         prior_precision_retention            prior precision (inverse variance) of retention
    %         prior_precision_drift                prior precision (inverse variance) of drift
    %         gamma_context                        gamma hyperparameter of the Chinese restaurant franchise for the context transitions
    %         alpha_context                        alpha hyperparameter of the Chinese restaurant franchise for the context transitions
    %         rho_context                          rho (self-transition) hyperparameter of the Chinese restaurant franchise for the context transitions
    %     parameters if cues are present
    %         gamma_cue                            gamma hyperparameter of the Chinese restaurant franchise for the cue emissions
    %         alpha_cue                            alpha hyperparameter of the Chinese restaurant franchise for the cue emissions
    %     parameters if inferring bias
    %         infer_bias                           infer the measurment bias (true) or not (false)
    %         prior_precision_bias                 precision (inverse variance) of prior of measurement bias
    %     paradigm
    %         perturbations                        vector of perturbations (NaN on channel trials)
    %         cues                                 vector of sensory cues (encode cues using integers starting from 1)
    %         eraser_trials                        trials on which to replace predicted probabilities with stationary probabilities
    %     runs
    %         runs                                 number of runs, each conditioned on a different observation noise sequence
    %     parallel processing of runs
    %         max_cores                            maximum number of CPU cores available (0 implements serial processing of runs)
    %     model implementation
    %         particles                            number of particles
    %         max_contexts                         maximum number of contexts that can be instantiated
    %     measured adaptation data
    %         adaptation                           vector of adaptation data (NaN if adaptation not measured on a trial)
    %     store
    %         store                                variables to store in memory
    %     plot flags
    %         plot_state_given_context             plot state | context
    %         plot_predicted_probabilities         plot predicted probabilities
    %         plot_responsibilities                plot responsibilities (including novel context probability)
    %         plot_stationary_probabilities        plot stationary probabilities
    %         plot_retention_given_context         plot retention | context
    %         plot_drift_given_context             plot drift | context
    %         plot_bias_given_context              plot bias | context
    %         plot_global_transition_probabilities plot global transition probabilities
    %         plot_local_transition_probabilities  plot local transition probabilities
    %         plot_global_cue_probabilities        plot global cue probabilities
    %         plot_local_cue_probabilities         plot local cue probabilities
    %         plot_state                           plot state (marginal distribution)
    %         plot_average_state                   plot average state (marginal distribution)
    %         plot_bias                            plot bias (marginal distribution)
    %         plot_average_bias                    plot average bias (marginal distribution)
    %         plot_state_feedback                  plot predicted state feedback (marginal distribution)
    %         plot_explicit_component              plot explicit component of learning
    %         plot_implicit_component              plot implicit component of learning
    %     plot inputs
    %         retention_values                     specify values to evaluate p(retention) at if plot_retention_given_context == true
    %         drift_values                         specify values to evaluate p(drift) at if plot_drift_given_context == true
    %         state_values                         specify values to evaluate p(state) at if plot_state_given_context == true or plot_state == true
    %         bias_values                          specify values to evaluate p(bias) at if plot_bias_given_context == true or plot_bias == true
    %         state_feedback_values                specify values to evaluate p(state feedback) at if plot_state_feedback == true
    %     miscellaneous user data
    %         user_data                            any data the user would like to associate with an object of the class
    %
    % VARIABLES                                    description
    %     stored before resampling (so that they do not depend on the state feedback)
    %         cPred                                predicted context probabilities (conditioned on the cue)
    %         pPred                                variance of the predictive distribution of the state feedback in each context
    %         vPred                                variance of the predictive distribution of the state in each context 
    %         xPred                                mean of the predictive distribution of the state in each context
    %         yPred                                mean of the predictive distribution of the state feedback in each context
    %     stored after resampling
    %         a                                    state retention factor in each context
    %         adCovar                              covariance of the posterior of the retention and drift in each context
    %         adMu                                 mean of the posterior of the retention and drift in each context
    %         adSS1                                sufficient statistic #1 for the retention and drift parameters in each context
    %         adSS2                                sufficient statistic #2 for the retention and drift parameters in each context
    %         b                                    bias in each context
    %         beta                                 global transition distribution
    %         betaE                                global cue distribution  
    %         betaEPosterior                       parameters of the posterior of the global cue distribution                                      
    %         betaPosterior                        parameters of the posterior of the global transition distribution                           
    %         bMu                                  mean of the posterior of the bias in each context
    %         bPredMarg                            bias (marginal distribution, discretised)
    %         bSS1                                 sufficient statistic #1 for the bias parameter in each context
    %         bSS2                                 sufficient statistic #2 for the bias parameter in each context
    %         bVar                                 variance of the posterior of the bias in each context
    %         C                                    number of instantiated contexts
    %         c                                    sampled context
    %         cFilt                                context responsibilities
    %         cInf                                 stationary context probabilities
    %         cPrev                                context sampled on the previous trial
    %         cPrior                               prior context probabilities (not conditioned on the cue)
    %         d                                    drift in each context
    %         e                                    state feedback prediction error in each context
    %         emissionMatrix                       expected value of the cue probability matrix
    %         explicit                             explicit component of learning
    %         implicit                             implicit component of learning
    %         iResamp                              indices of resampled particles
    %         iX                                   index of the observed state
    %         k                                    Kalman gain in each context
    %         m                                    number of tables in restaurant i serving dish j (Chinese restaurant franchise for the context transitions)
    %         mE                                   number of tables in restaurant i serving dish j (Chinese restaurant franchise for the cue emissions)
    %         motorNoise                           motor noise
    %         n                                    context transition counts
    %         nE                                   cue emission counts
    %         pQ                                   probability of the observed cue in each context        
    %         pY                                   probability of the observed state feedback in each context
    %         Q                                    number of cues observed
    %         sensoryNoise                         sensory noise
    %         transitionMatrix                     expected value of the context transition probability matrix
    %         vFilt                                variance of the filtered distribution of the state in each context 
    %         vPrev                                variance of the filtered distribution of the state in each context on previous trial
    %         xFilt                                mean of the filtered distribution of the state in each context 
    %         xHat                                 average predicted state (average across contexts and particles)
    %         xPredMarg                            predicted state (marginal distribution, discretised)
    %         xPrev                                mean of the filtered distribution of the state in each context on previous trial
    %         xPrevSamp                            states sampled on previous trial (to update the sufficient statistics for the retention and drift parameters in each context)
    %         xSamp                                state sampled on current trial (to update the sufficient statistics for the bias parameter in each context)
    %         xSampOld                             states sampled on current trial (to update the sufficient statistics for the retention and drift parameters in each context)
    %         yHat                                 average predicted state feedback (average across contexts and particles)
    %         yPredMarg                            predicted state feedback (marginal distribution, discretised)
    
    properties
        
        % core parameters - values taken from Heald et al. (2020) Table S1 (A)
        sigma_process_noise = 0.0089 
        sigma_sensory_noise = 0.03
        sigma_motor_noise = 0.0182
        prior_mean_retention = 0.9425 
        prior_precision_retention = 837.1539^2
        prior_precision_drift = 1.2227e+03^2
        gamma_context = 0.1
        alpha_context = 8.9556
        rho_context = 0.2501
        
        % parameters if cues are present
        gamma_cue = 0.1
        alpha_cue = 0.1
        
        % parameters if inferring a bias
        infer_bias = false
        prior_precision_bias = 70^2
        
        % paradigm
        perturbations
        cues
        eraser_trials
        
        % number of runs
        runs = 1
        
        % parallel processing
        max_cores = 0
        
        % model implementation
        particles = 100
        max_contexts = 10
        
        % measured adaptation data
        adaptation
        
        % store
        store = {}
        
        % plot flags
        plot_state_given_context = false
        plot_predicted_probabilities = false
        plot_responsibilities = false
        plot_stationary_probabilities = false
        plot_retention_given_context = false
        plot_drift_given_context = false
        plot_bias_given_context = false
        plot_global_transition_probabilities = false
        plot_local_transition_probabilities = false
        plot_global_cue_probabilities = false
        plot_local_cue_probabilities = false
        plot_state = false
        plot_average_state = false
        plot_bias = false
        plot_average_bias = false
        plot_state_feedback = false
        plot_explicit_component = false
        plot_implicit_component = false
        
        % plot inputs
        retention_values = linspace(0.8,1,500);
        drift_values = linspace(-0.1,0.1,500);
        state_values = linspace(-1.5,1.5,500);
        bias_values = linspace(-1.5,1.5,500);
        state_feedback_values = linspace(-1.5,1.5,500);
        
        % user data
        user_data

    end

    methods
        
        function S = simulate_COIN(obj)
            
            % set properties based on the vaues of other properties
            obj = set_property_values(obj);
            
            % number of trials
            T = numel(obj.perturbations);
            
            % preallocate memory
            tmp = cell(1,obj.runs);
            
            if isempty(obj.adaptation)
                
                trials = 1:T;
                
                % perform obj.runs runs
                parfor (run = 1:obj.runs,obj.max_cores)
                    fprintf('simulating the COIN model\n')
                    tmp{run} = obj.main_loop(trials).stored;
                end
                
                % assign equal weights to all runs
                w = ones(1,obj.runs)/obj.runs;

            else
                
                % perform obj.runs runs
                % resample runs if the effective sample size falls below threshold

                % preallocate memory
                D_in = cell(1,obj.runs);
                D_out = cell(1,obj.runs);
                
                % initialise weights to be uniform
                w = ones(1,obj.runs)/obj.runs;

                % effective sample size threshold for resampling
                thresholdESS = 0.5*obj.runs;

                % trials on which adaptation was measured
                aT = find(~isnan(obj.adaptation));

                % simulate trials inbetween trials on which adaptation was measured
                for i = 1:numel(aT)

                    if i == 1
                        trials = 1:aT(i);
                        fprintf('simulating the COIN model from trial 1 to trial %d\n',aT(i))
                    else
                        trials = aT(i-1)+1:aT(i);
                        fprintf('simulating the COIN model from trial %d to trial %d\n',aT(i-1)+1,aT(i))
                    end

                    parfor (run = 1:obj.runs,obj.max_cores)
                        if i == 1
                            D_out{run} = obj.main_loop(trials);
                        else
                            D_out{run} = obj.main_loop(trials,D_in{run});
                        end
                    end

                    % calculate the log likelihood
                    logLikelihood = zeros(1,obj.runs);
                    for run = 1:obj.runs
                        error = D_out{run}.stored.yHat(aT(i)) - obj.adaptation(aT(i));
                        logLikelihood(run) = -(log(2*pi*obj.sigma_motor_noise^2) + (error/obj.sigma_motor_noise).^2)/2; 
                    end

                    % update the weights and normalise
                    lw = logLikelihood + log(w);
                    lw = lw - obj.log_sum_exp(lw');
                    w = exp(lw);

                    % calculate the effective sample size
                    ESS = 1/(sum(w.^2));

                    % if the effective sample size falls below thresholdESS, resample
                    if ESS < thresholdESS
                        fprintf('effective sample size = %.1f --- resampling\n',ESS)
                        iResamp = obj.systematic_resampling(w);
                        for run = 1:obj.runs
                            D_in{run} = D_out{iResamp(run)};
                        end
                        w = ones(1,obj.runs)/obj.runs;
                    else
                        fprintf('effective sample size = %.1f --- not resampling\n',ESS)
                        D_in = D_out;
                    end

                end
                
                if aT(end) == T
                    
                    for run = 1:obj.runs
                        tmp{run} = D_in{run}.stored;
                    end

                elseif aT(end) < T

                    % simulate to the last trial
                    fprintf('simulating the COIN model from trial %d to trial %d\n',aT(end)+1,T)

                    trials = aT(end)+1:T;

                    parfor (run = 1:obj.runs,obj.max_cores)
                        tmp{run} = obj.main_loop(trials,D_in{run}).stored;
                    end
                    
                end

            end
            
            % preallocate memory
            S.runs = cell(1,obj.runs);
            
            % assign data to S
            for run = 1:obj.runs
                S.runs{run} = tmp{run};
            end
            S.weights = w;
            S.properties = obj;
            
        end
        
        function obj = set_property_values(obj)

            % specify variables that need to be stored for plots
            tmp = {};
            if obj.plot_state_given_context
                if isempty(obj.state_values)
                    error('Specify points at which to evaluate the state | context distribution. Use property ''state_values''.')
                end
                tmp = cat(2,tmp,{'xPred','vPred'});
            end
            if obj.plot_predicted_probabilities
                tmp = cat(2,tmp,'cPred');
            end
            if obj.plot_responsibilities
                tmp = cat(2,tmp,'cFilt');
            end
            if obj.plot_stationary_probabilities
                tmp = cat(2,tmp,'cInf');
            end
            if obj.plot_retention_given_context
                if isempty(obj.retention_values)
                    error('Specify points at which to evaluate the retention | context distribution. Use property ''retention_values''.')
                end
                tmp = cat(2,tmp,{'adMu','adCovar'});
            end
            if obj.plot_drift_given_context
                if isempty(obj.drift_values)
                    error('Specify points at which to evaluate the drift | context distribution. Use property ''drift_values''.')
                end
                tmp = cat(2,tmp,{'adMu','adCovar'});
            end
            if obj.plot_bias_given_context
                if obj.infer_bias
                    if isempty(obj.bias_values)
                        error('Specify points at which to evaluate the bias | context distribution. Use property ''bias_values''.')
                    end
                    tmp = cat(2,tmp,{'bMu','bVar'});
                else
                    error('You must learn the measurement bias parameter to use plot_bias_given_context. Set property ''infer_bias'' to true.')
                end
            end
            if obj.plot_global_transition_probabilities
                tmp = cat(2,tmp,'betaPosterior');
            end
            if obj.plot_local_transition_probabilities
                tmp = cat(2,tmp,'transitionMatrix');
            end
            if obj.plot_local_cue_probabilities
                if isempty(obj.cues) 
                    error('An experiment must have sensory cues to use plot_local_cue_probabilities.')
                else
                    tmp = cat(2,tmp,'emissionMatrix');
                end
            end
            if obj.plot_global_cue_probabilities
                if isempty(obj.cues) 
                    error('An experiment must have sensory cues to use plot_global_cue_probabilities.')
                else
                    tmp = cat(2,tmp,'betaEPosterior');
                end
            end
            if obj.plot_state
                if isempty(obj.state_values)
                    error('Specify points at which to evaluate the state distribution. Use property ''state_values''.')
                end
                tmp = cat(2,tmp,'xPredMarg','xHat');
            end
            if obj.plot_average_state
                tmp = cat(2,tmp,'xHat');
            end
            if obj.plot_bias
                if isempty(obj.bias_values)
                    error('Specify points at which to evaluate the bias distribution. Use property ''bias_values''.')
                end
                if obj.infer_bias
                    tmp = cat(2,tmp,'bPredMarg','implicit');
                else
                    error('You must learn the measurement bias parameter to use plot_bias. Set property ''infer_bias'' to true.')
                end
            end
            if obj.plot_average_bias
                tmp = cat(2,tmp,'implicit');
            end
            if obj.plot_state_feedback
                if isempty(obj.state_feedback_values)
                    error('Specify points at which to evaluate the state feedback distribution. Use property ''state_feedback_values''.')
                end
                tmp = cat(2,tmp,'yPredMarg');
            end
            if obj.plot_explicit_component
                tmp = cat(2,tmp,'explicit');
            end
            if obj.plot_implicit_component
                tmp = cat(2,tmp,'implicit');
            end
            if ~isempty(tmp)
                tmp = cat(2,tmp,{'c','iResamp'});
            end
            tmp = cat(2,tmp,{'y','yHat'}); % always store the state feedback and model output
            
            % add strings in tmp to the store property of obj
            for i = 1:numel(tmp)
                if ~any(strcmp(obj.store,tmp{i}))
                    obj.store{end+1} = tmp{i};
                end
            end
            
            if ~isempty(obj.adaptation) && numel(obj.adaptation) ~= numel(obj.perturbations)
                error('Property ''adaptation'' should be a vector with one element per trial (use NaN on trials where adaptation was not measured).')
            end
            
        end
        
        function objective = objective_COIN(obj)
            
            S = numel(obj); % number of participants
            n = sum(~isnan(obj(1).adaptation)); % number of adaptation measurements per participant
            aT = zeros(n,S);
            data = zeros(n,S);
            for s = 1:S

                obj(s) = set_property_values(obj(s));

                if isrow(obj(s).adaptation)
                    obj(s).adaptation = obj(s).adaptation';
                end
                
                % trials on which adaptation was measured
                aT(:,s) = find(~isnan(obj(s).adaptation));

                % measured adaptation
                data(:,s) = obj(s).adaptation(aT(:,s));
                
            end

            logLikelihood = zeros(obj(1).runs,1);
            parfor (run = 1:obj(1).runs,obj(1).max_cores)

                model = zeros(n,S);
                for s = 1:S
                    
                    % number of trials
                    T = numel(obj(s).perturbations);
                    
                    trials = 1:T;

                    % model adaptation
                    model(:,s) = obj(s).main_loop(trials).stored.yHat(aT(:,s));
                
                end

                % error between average model adaptation and average measured adaptation
                error = mean(model-data,2);

                % log likelihood (probability of data given parameters)
                logLikelihood(run) = sum(-(log(2*pi*obj(1).sigma_motor_noise^2/S) + error.^2/(obj(1).sigma_motor_noise.^2/S))/2); % variance scaled by the number of participants

            end

            % negative of the log of the average likelihood across runs
            objective = -(log(1/obj(1).runs) + obj(1).log_sum_exp(logLikelihood));
        
        end

        
        function D = main_loop(obj,trials,varargin)

            if trials(1) == 1
                D = obj.initialise_COIN;                                % initialise the model
            else
                D = varargin{1};
            end
            for trial = trials
                D.t = trial;                                            % set the current trial number
                D = obj.predict_context(D);                             % predict the context
                D = obj.predict_states(D);                              % predict the states
                D = obj.predict_state_feedback(D);                      % predict the state feedback
                D = obj.resample_particles(D);                          % resample particles
                D = obj.sample_context(D);                              % sample the context
                D = obj.update_belief_about_states(D);                  % update the belief about the states given state feedback
                D = obj.sample_states(D);                               % sample the states
                D = obj.update_sufficient_statistics_for_parameters(D); % update the sufficient statistics for the parameters
                D = obj.sample_parameters(D);                           % sample the parameters
                D = obj.store_variables(D);                             % store variables for analysis if desired
            end
            
        end
        
        function D = initialise_COIN(obj)
            
            % number of trials
            D.T = numel(obj.perturbations);
            
            % treat channel trials as non observations with respect to the state feedback
            D.feedback = ones(1,D.T);
            D.feedback(isnan(obj.perturbations)) = 0;
            
            % self-transition bias
            D.kappa = obj.alpha_context*obj.rho_context/(1-obj.rho_context);
            
            % observation noise standard deviation
            D.sigma_observation_noise = sqrt(obj.sigma_sensory_noise^2 + obj.sigma_motor_noise^2);
            
            % matrix of context-dependent observation vectors
            D.H = eye(obj.max_contexts+1);
            
            % do cues exist?
            if isempty(obj.cues) 
                D.cuesExist = 0;
            else
                D.cuesExist = 1;
            end

            % current trial
            D.t = 0;

            % number of contexts instantiated so far
            D.C = zeros(1,obj.particles);

            % context transition counts
            D.n = zeros(obj.max_contexts+1,obj.max_contexts+1,obj.particles);

            % sampled context
            D.c = ones(1,obj.particles); % treat trial 1 as a (context 1) self transition
                
            if D.cuesExist
                
                D = obj.check_cue_labels(D);
                
                % number of contextual cues observed so far
                D.Q = 0;
                
                % cue emission counts
                D.nE = zeros(obj.max_contexts+1,max(obj.cues)+1,obj.particles);
            end

            % sufficient statistics for the parameters of the state dynamics
            % function
            D.adSS1 = zeros(obj.max_contexts+1,obj.particles,2);
            D.adSS2 = zeros(obj.max_contexts+1,obj.particles,2,2);

            % sufficient statistics for the parameters of the observation function
            D.bSS1 = zeros(obj.max_contexts+1,obj.particles);
            D.bSS2 = zeros(obj.max_contexts+1,obj.particles);

            % sample parameters from the prior
            D = sample_parameters(obj,D);
            
            % mean and variance of state (stationary distribution)
            D.xFilt = D.d./(1-D.a);
            D.vFilt = obj.sigma_process_noise^2./(1-D.a.^2);
            
        end
        
        function D = predict_context(obj,D)
            
            if ismember(D.t,obj.eraser_trials)
                % if some event (e.g. a working memory task) causes the context 
                % probabilities to be erased, set them to their stationary values
                for particle = 1:obj.particles
                    C = sum(D.transitionMatrix(:,1,particle)>0);
                    T = D.transitionMatrix(1:C,1:C,particle);        
                    D.cPrior(1:C,particle) = obj.stationary_distribution(T);
                end
            else
                i = sub2ind(size(D.transitionMatrix),repmat(D.c,[obj.max_contexts+1,1]),repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.particles,[obj.max_contexts+1,1]));
                D.cPrior = D.transitionMatrix(i);
            end

            if D.cuesExist
                i = sub2ind(size(D.emissionMatrix),repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(obj.cues(D.t),[obj.max_contexts+1,obj.particles]),repmat(1:obj.particles,[obj.max_contexts+1,1]));
                D.pQ = D.emissionMatrix(i);
                D.cPred = D.cPrior.*D.pQ;
                D.cPred = D.cPred./sum(D.cPred,1);
            else
                D.cPred = D.cPrior;
            end
            
        end

        function D = predict_states(obj,D)

            % propagate states
            D.xPred = D.a.*D.xFilt + D.d;
            D.vPred = D.a.*D.vFilt.*D.a + obj.sigma_process_noise^2;

            % index of novel states
            iNewX = sub2ind([obj.max_contexts+1,obj.particles],D.C+1,1:obj.particles);

            % novel states are distributed according to the stationary distribution
            D.xPred(iNewX) = D.d(iNewX)./(1-D.a(iNewX));
            D.vPred(iNewX) = obj.sigma_process_noise^2./(1-D.a(iNewX).^2);

            % predict state (marginalise over contexts and particles)
            % mean of distribution
            D.xHat = sum(D.cPred.*D.xPred,'all')/obj.particles;

            if any(strcmp(obj.store,'explicit'))
                if D.t == 1
                    i = ones(1,obj.particles);
                else
                    [~,i] = max(D.cFilt,[],1);
                end
                i = sub2ind(size(D.xPred),i,1:obj.particles);
                D.explicit = mean(D.xPred(i));
            end

        end
        
        function D = predict_state_feedback(obj,D)  

            % predict state feedback for each context
            D.yPred = D.xPred + D.b;

            % variance of state feedback prediction for each context
            D.pPred = D.vPred + D.sigma_observation_noise^2;
            
            D = obj.compute_marginal_distribution(D);

            % predict state feedback (marginalise over contexts and particles)
            % mean of distribution
            D.yHat = sum(D.cPred.*D.yPred,'all')/obj.particles;

            if any(strcmp(obj.store,'implicit'))
                D.implicit = D.yHat - D.xHat;
            end

            % sensory and motor noise
            D.sensoryNoise = obj.sigma_sensory_noise*randn;
            D.motorNoise = obj.sigma_motor_noise*randn;

            % state feedback
            D.y = obj.perturbations(D.t) + D.sensoryNoise + D.motorNoise;

            % state feedback prediction error
            D.e = D.y - D.yPred;

        end
        
        function D = resample_particles(obj,D)

            D.pY = normpdf(D.y,D.yPred,sqrt(D.pPred)); % p(y_t|c_t)

            if D.feedback(D.t)
                if D.cuesExist
                    pC = log(D.cPrior) + log(D.pQ) + log(D.pY); % log p(y_t,q_t,c_t)
                else
                    pC = log(D.cPrior) + log(D.pY); % log p(y_t,c_t)
                end
            else
                if D.cuesExist
                    pC = log(D.cPrior) + log(D.pQ);% log p(q_t,c_t)
                else
                    pC = log(D.cPrior); % log p(c_t)
                end
            end

            lW = obj.log_sum_exp(pC); % log p(y_t,q_t)

            pC = pC - lW; % log p(c_t|y_t,q_t)

            % weights for resampling
            w = exp(lW - obj.log_sum_exp(lW'));

            % draw indices of particles to propagate
            if D.feedback(D.t) || D.cuesExist
                D.iResamp = obj.systematic_resampling(w);
            else
                D.iResamp = 1:obj.particles;
            end
            
            % optionally store predictive variables
            % n.b. these are stored before resampling (so that they do not condition on the current observations)
            for i = 1:numel(obj.store)
                if contains(obj.store{i},'Pred')
                    if D.t == 1
                        % preallocate memory for variables to be stored
                        s = [eval(sprintf('size(D.%s)',obj.store{i})) D.T];
                        eval(sprintf('D.stored.%s = squeeze(zeros(s));',obj.store{i}))
                    end
                    % store variables
                    dims = eval(sprintf('sum(size(D.%s)>1)',obj.store{i}));
                    eval(sprintf('D.stored.%s(%sD.t) = D.%s;',obj.store{i},repmat(':,',[1,dims]),obj.store{i}))
                end
            end

            % resample variables (particles)
            D.cPrev = D.c(D.iResamp);
            D.cPrior = D.cPrior(:,D.iResamp);
            D.cFilt = exp(pC(:,D.iResamp)); % p(c_t|y_t,q_t)
            D.C = D.C(D.iResamp);
            D.xPred = D.xPred(:,D.iResamp);
            D.vPred = D.vPred(:,D.iResamp);
            D.e = D.e(:,D.iResamp);
            D.pPred = D.pPred(:,D.iResamp);
            D.pY = D.pY(:,D.iResamp);
            D.beta = D.beta(:,D.iResamp);
            D.n = D.n(:,:,D.iResamp);
            D.xPrev = D.xFilt(:,D.iResamp);
            D.vPrev = D.vFilt(:,D.iResamp);

            if D.cuesExist 
                D.betaE  = D.betaE(:,D.iResamp);
                D.nE = D.nE(:,:,D.iResamp);
            end

            D.a = D.a(:,D.iResamp);
            D.d = D.d(:,D.iResamp);

            D.adSS1 = D.adSS1(:,D.iResamp,:);
            D.adSS2 = D.adSS2(:,D.iResamp,:,:);

            if obj.infer_bias
                D.bSS1 = D.bSS1(:,D.iResamp);
                D.bSS2 = D.bSS2(:,D.iResamp);
            end

        end

        function D = sample_context(obj,D)

            % sample the context
            D.c = sum(rand(1,obj.particles) > cumsum(D.cFilt),1) + 1;

            % incremement the context count
            D.pNewX = find(D.c > D.C);
            D.pOldX = find(D.c <= D.C);
            D.C(D.pNewX) = D.C(D.pNewX) + 1;

            pBetaX = D.pNewX(D.C(D.pNewX) ~= obj.max_contexts);
            iBu = sub2ind([obj.max_contexts+1,obj.particles],D.c(pBetaX),pBetaX);

            % sample the next stick-breaking weight
            beta = betarnd(1,obj.gamma_context*ones(1,numel(pBetaX)));

            % update the global transition distribution
            D.beta(iBu+1) = D.beta(iBu).*(1-beta);
            D.beta(iBu) = D.beta(iBu).*beta;

            if D.cuesExist
                if obj.cues(D.t) > D.Q 
                    % increment the cue count
                    D.Q = D.Q + 1;

                    % sample the next stick-breaking weight
                    beta = betarnd(1,obj.gamma_cue*ones(1,obj.particles));

                    % update the global cue distribution
                    D.betaE(D.Q+1,:) = D.betaE(D.Q,:).*(1-beta);
                    D.betaE(D.Q,:) = D.betaE(D.Q,:).*beta;
                end

            end

        end

        function D = update_belief_about_states(obj,D)

            D.k = D.vPred./D.pPred;
            if D.feedback(D.t)
                D.xFilt = D.xPred + D.k.*D.e.*D.H(D.c,:)';
                D.vFilt = (1 - D.k.*D.H(D.c,:)').*D.vPred;
            else
                D.xFilt = D.xPred;
                D.vFilt = D.vPred;
            end

        end
        
        function D = sample_states(obj,D)

            nNewX = numel(D.pNewX);
            iOldX = sub2ind([obj.max_contexts+1,obj.particles],D.c(D.pOldX),D.pOldX);
            iNewX = sub2ind([obj.max_contexts+1,obj.particles],D.c(D.pNewX),D.pNewX);

            % for states that have been observed before, sample x_{t-1}, and then sample x_{t} given x_{t-1}

                % sample x_{t-1} using a fixed-lag (lag 1) forward-backward smoother
                g = D.a.*D.vPrev./D.vPred;
                m = D.xPrev + g.*(D.xFilt - D.xPred);
                v = D.vPrev + g.*(D.vFilt - D.vPred).*g;
                D.xPrevSamp = m + sqrt(v).*randn(obj.max_contexts+1,obj.particles);

                % sample x_t conditioned on x_{t-1} and y_t
                if D.feedback(D.t)
                    w = (D.a.*D.xPrevSamp + D.d)./obj.sigma_process_noise^2 + D.H(D.c,:)'./D.sigma_observation_noise^2.*(D.y - D.b);
                    v = 1./(1./obj.sigma_process_noise^2 + D.H(D.c,:)'./D.sigma_observation_noise^2);
                else
                    w = (D.a.*D.xPrevSamp + D.d)./obj.sigma_process_noise^2;
                    v = 1./(1./obj.sigma_process_noise^2);
                end
                D.xSampOld = v.*w + sqrt(v).*randn(obj.max_contexts+1,obj.particles);

            % for novel states, sample x_t from the filtering distribution
            
                xSampNew = D.xFilt(iNewX) + sqrt(D.vFilt(iNewX)).*randn(1,nNewX);

            D.xSamp = [D.xSampOld(iOldX) xSampNew];
            D.iX = [iOldX iNewX];

        end
        
        function D = update_sufficient_statistics_for_parameters(obj,D)

            % update the sufficient statistics for the parameters of the 
            % context transition probabilities
            D = obj.update_sufficient_statistics_pi(D);

            % update the sufficient statistics for the parameters of the 
            % cue probabilities
            if D.cuesExist 
                D = obj.update_sufficient_statistics_phi(D);
            end

            if D.t > 1
                % update the sufficient statistics for the parameters of the 
                % state dynamics function
                D = obj.update_sufficient_statistics_ad(D);
            end

            % update the sufficient statistics for the parameters of the 
            % observation function
            if obj.infer_bias && D.feedback(D.t)
                D = obj.update_sufficient_statistics_b(D);
            end

        end
         
        function D = sample_parameters(obj,D)

            % sample beta
            D = obj.sample_beta(D);

            % update context transition matrix
            D = obj.update_transition_matrix(D);

            if D.cuesExist 
                % sample betaE
                D = obj.sample_betaE(D);

                % update cue probability matrix
                D = obj.update_emission_matrix(D);
            end

            % sample the parameters of the state dynamics function
            D = obj.sample_ad(D);

            % sample the parameters of the observation function
            if obj.infer_bias
                D = obj.sample_b(D);
            else
                D.b = 0;
            end
            
        end
         
        function D = store_variables(obj,D)
             
            % optionally store filtered variables
            for i = 1:numel(obj.store)
                if ~contains(obj.store{i},'Pred')
                    if D.t == 1
                        % preallocate memory for variables to be stored
                        s = [eval(sprintf('size(D.%s)',obj.store{i})) D.T];
                        eval(sprintf('D.stored.%s = squeeze(zeros(s));',obj.store{i}))
                    end
                    % store variables
                    dims = eval(sprintf('sum(size(D.%s)>1)',obj.store{i}));
                    eval(sprintf('D.stored.%s(%sD.t) = D.%s;',obj.store{i},repmat(':,',[1,dims]),obj.store{i}))
                end
            end
            
        end
         
        function D = update_sufficient_statistics_ad(obj,D)

            % augment the state vector: x_{t-1} --> [x_{t-1}; 1]
            xa = ones(obj.max_contexts+1,obj.particles,2);
            xa(:,:,1) = D.xPrevSamp;

            % identify states that are not novel
            I = reshape(sum(D.n,2),[obj.max_contexts+1,obj.particles]) > 0;

            SS = D.xSampOld.*xa; % x_t*[x_{t-1}; 1]
            D.adSS1 = D.adSS1 + SS.*I;

            SS = reshape(xa,[obj.max_contexts+1,obj.particles,2]).*reshape(xa,[obj.max_contexts+1,obj.particles,1,2]); % [x_{t-1}; 1]*[x_{t-1}; 1]'
            D.adSS2 = D.adSS2 + SS.*I;

        end
         
        function D = update_sufficient_statistics_b(obj,D)

            D.bSS1(D.iX) = D.bSS1(D.iX) + (D.y - D.xSamp); % y_t - x_t
            D.bSS2(D.iX) = D.bSS2(D.iX) + 1; % 1(c_t = j)
            
        end

        function D = update_sufficient_statistics_phi(obj,D)

            i = sub2ind([obj.max_contexts+1,max(obj.cues)+1,obj.particles],D.c,obj.cues(D.t)*ones(1,obj.particles),1:obj.particles); % 1(c_t = j, q_t = k)
            D.nE(i) = D.nE(i) + 1;   

        end
         
        function D = update_sufficient_statistics_pi(obj,D)

            i = sub2ind([obj.max_contexts+1,obj.max_contexts+1,obj.particles],D.cPrev,D.c,1:obj.particles); % 1(c_{t-1} = i, c_t = j)
            D.n(i) = D.n(i) + 1;
            
        end
         
        function D = sample_ad(obj,D)
            
            % prior mean and precision matrix
            adMu = [obj.prior_mean_retention 0]';
            adLambda = diag([obj.prior_precision_retention obj.prior_precision_drift]);

            % update the parameters of the posterior
            D.adCovar = obj.per_slice_invert(adLambda + permute(reshape(D.adSS2,[(obj.max_contexts+1)*obj.particles,2,2]),[2,3,1])/obj.sigma_process_noise^2);
            D.adMu = obj.per_slice_multiply(D.adCovar,adLambda*adMu + reshape(D.adSS1,[(obj.max_contexts+1)*obj.particles,2])'/obj.sigma_process_noise^2);
            
            % sample the parameters of the state dynamics function
            ad = obj.sample_from_truncated_bivariate_normal(D.adMu,D.adCovar);
            D.a = reshape(ad(1,:),[obj.max_contexts+1,obj.particles]);
            D.d = reshape(ad(2,:),[obj.max_contexts+1,obj.particles]);
            
            % reshape
            D.adMu = reshape(D.adMu,[2,obj.max_contexts+1,obj.particles]);
            D.adCovar = reshape(D.adCovar,[2,2,obj.max_contexts+1,obj.particles]);

        end
         
        function D = sample_b(obj,D)
            
            % prior mean
            bMu = 0;

            % update the parameters of the posterior
            D.bVar = 1./(obj.prior_precision_bias + D.bSS2./D.sigma_observation_noise^2);
            D.bMu = D.bVar.*(obj.prior_precision_bias.*bMu + D.bSS1./D.sigma_observation_noise^2);

            % sample the parameters of the observation function
            D.b = obj.sample_from_univariate_normal(D.bMu,D.bVar);

        end
         
        function D = sample_beta(obj,D)

            if D.t == 0
                % global transition distribution
                D.beta = zeros(obj.max_contexts+1,obj.particles);
                D.beta(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.m = randnumtable(permute(obj.alpha_context*D.beta,[3,1,2]) + D.kappa*eye(obj.max_contexts+1),D.n);

                % sample the number of tables in restaurant i considering dish j
                barM = D.m;
                if obj.rho_context > 0
                    i = sub2ind([obj.max_contexts+1,obj.max_contexts+1,obj.particles],repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.particles,[obj.max_contexts+1,1]));
                    p = obj.rho_context./(obj.rho_context + D.beta(D.m(i) ~= 0)*(1-obj.rho_context));
                    i = i(D.m(i) ~= 0);
                    barM(i) = D.m(i) - randbinom(p,D.m(i));
                end
                barM(1,1,(barM(1,1,:) == 0)) = 1;

                % sample beta
                i = find(D.C ~= obj.max_contexts);
                iBu = sub2ind([obj.max_contexts+1,obj.particles],D.C(i)+1,i);
                D.betaPosterior = squeeze(sum(barM,1));
                D.betaPosterior(iBu) = obj.gamma_context;
                D.beta = obj.sample_from_dirichlet(D.betaPosterior);
            end

        end
         
        function D = sample_betaE(obj,D)

            if D.t == 0
                % global cue distribution
                D.betaE = zeros(max(obj.cues)+1,obj.particles);
                D.betaE(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.mE = randnumtable(repmat(permute(obj.alpha_cue.*D.betaE,[3,1,2]),[obj.max_contexts+1,1,1]),D.nE);

                % sample betaE
                D.betaEPosterior = reshape(sum(D.mE,1),[max(obj.cues)+1,obj.particles]);
                D.betaEPosterior(D.Q+1,:) = obj.gamma_cue;
                D.betaE = obj.sample_from_dirichlet(D.betaEPosterior);
            end

        end
         
        function D = update_emission_matrix(obj,D)

            D.emissionMatrix = reshape(obj.alpha_cue.*D.betaE,[1,max(obj.cues)+1,obj.particles]) + D.nE;
            D.emissionMatrix = D.emissionMatrix./sum(D.emissionMatrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.beta,[obj.max_contexts+1,1,obj.particles]) > 0;
            D.emissionMatrix = D.emissionMatrix.*I;

        end
        
        function D = update_transition_matrix(obj,D)

            D.transitionMatrix = reshape(obj.alpha_context*D.beta,[1,obj.max_contexts+1,obj.particles]) + D.n + D.kappa*eye(obj.max_contexts+1);
            D.transitionMatrix = D.transitionMatrix./sum(D.transitionMatrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.beta,[obj.max_contexts+1,1,obj.particles]) > 0;
            D.transitionMatrix = D.transitionMatrix.*I;

            % compute stationary context probabilities if required
            if any(strcmp(obj.store,'cInf')) && D.t > 0
                D.cInf = zeros(obj.max_contexts+1,obj.particles);
                for particle = 1:obj.particles
                    C = D.C(particle);
                    T = D.transitionMatrix(1:C+1,1:C+1,particle);
                    D.cInf(1:C+1,particle) = obj.stationary_distribution(T);
                end
            end

        end
         
        function D = check_cue_labels(obj,D)

            % check cues are numbered according to the order they were presented in the experiment
            for trial = 1:D.T
                if trial == 1
                    if ~eq(obj.cues(trial),1)
                        D = obj.renumber_cues(D);
                        break
                    end
                else
                    if ~ismember(obj.cues(trial),obj.cues(1:trial-1)) && ~eq(obj.cues(trial),max(obj.cues(1:trial-1))+1)
                        D = obj.renumber_cues(D);
                        break
                    end
                end
            end
            
        end

        function D = renumber_cues(obj,D)

            cueOrder = unique(obj.cues,'stable');
            if isrow(obj.cues)
                [obj.cues,~] = find(eq(obj.cues,cueOrder'));
            else
                [obj.cues,~] = find(eq(obj.cues,cueOrder')');
            end
            fprintf('Cues have been renumbered according to the order they were presented in the experiment.\n')

        end
         
        function D = compute_marginal_distribution(obj,D)
             
            if any(strcmp(obj.store,'xPredMarg'))
                % predict state (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.state_values,[1,1,numel(obj.state_values)]);
                mu = D.xPred;
                sd = sqrt(D.vPred);
                D.xPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end
            if any(strcmp(obj.store,'bPredMarg'))
                % predict bias (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.bias_values,[1,1,numel(obj.bias_values)]);
                mu = D.bMu;
                sd = sqrt(D.bVar);
                D.bPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end
            if any(strcmp(obj.store,'yPredMarg'))
                % predict state feedback (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.state_feedback_values,[1,1,numel(obj.state_feedback_values)]);
                mu = D.yPred;
                sd = sqrt(D.pPred);
                D.yPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end

        end
         
        function l = log_sum_exp(obj,logP)

            m = max(logP,[],1);
            l = m + log(sum(exp(logP - m),1));
            
        end
         
        function L = per_slice_cholesky(obj,V)

            % perform cholesky decomposition on each 2 x 2 slice of array V to
            % obtain lower triangular matrix L

            L = zeros(size(V));
            L(1,1,:) = sqrt(V(1,1,:));
            L(2,1,:) = V(2,1,:)./L(1,1,:);
            L(2,2,:) = sqrt(V(2,2,:) - L(2,1,:).^2);
            
        end

        function invL = per_slice_invert(obj,L)

            % invert each 2 x 2 slice of array L

            detL = L(1,1,:).*L(2,2,:)-L(1,2,:).*L(2,1,:);
            invL = [L(2,2,:) L(1,2,:); L(2,1,:) L(1,1,:)].*[1 -1; -1 1]./detL;
            
        end

        function C = per_slice_multiply(obj,A,B)

            % per slice matrix multiplication
            C = squeeze(sum(A.*permute(B,[3,1,2]),2));

        end
        
        function x = sample_from_dirichlet(obj,A)

            % sample categorical parameter from dirichlet distribution
            x = randgamma(A);
            x = x./sum(x,1);

        end
        
        function x = sample_from_truncated_bivariate_normal(obj,mu,V)

            % perform cholesky decomposition on the covariance matrix V
            cholV = obj.per_slice_cholesky(V);

            % truncation bounds for the state retention factor
            aMin = 0;
            aMax = 1;

            % equivalent truncation bounds for the standard normal distribution
            l = [(aMin-mu(1,:))./squeeze(cholV(1,1,:))'; -Inf*ones(1,size(V,3))];
            u = [(aMax-mu(1,:))./squeeze(cholV(1,1,:))'; Inf*ones(1,size(V,3))];

            % transform samples from the truncated standard normal distribution to
            % samples from the desired truncated normal distribution
            x = mu + obj.per_slice_multiply(cholV,reshape(trandn(l,u),[2,size(V,3)]));

        end

        function x = sample_from_univariate_normal(obj,mu,sigma)

            x = mu + sqrt(sigma).*randn(obj.max_contexts+1,obj.particles);

        end
        
        function p = stationary_distribution(obj,T)
            % stationary distribution of a time-homogeneous finite-state Markov chain.

            c = size(T,1);     % state-space cardinality (number of contexts)

            A = T'-eye(c);     % define A in Ax = b
            b = zeros(c,1);    % define b in Ax = b

            A(end+1,:) = 1;    % add normalisation constraint to ensure x is a valid
            b(end+1) = 1;      % probability distribution

            x = linsolve(A,b); % solve Ax = b for x

            x(x < 0) = 0;      % set any (slight) negative values to zero
            p = x'/sum(x);     % renormalise
            
        end

        function p = systematic_resampling(obj,w)

            % systematic resampling - O(n) complexity
            n = numel(w);
            Q = cumsum(w);
            y = linspace(0,1-1/n,n) + rand/n;
            p = zeros(n,1);
            i = 1; 
            j = 1;
            while i <= n && j <= n
                while Q(j) < y(i)
                    j = j + 1;
                end
                p(i) = j;
                i = i + 1;
            end
            
        end
        
        function P = plot_COIN(obj,S)
            
            [P,S,optAssignment,from_unique,cSeq,C] = obj.find_optimal_context_labels(S);
            
            [P,~] = obj.compute_variables_for_plotting(P,S,optAssignment,from_unique,cSeq,C);
            
            obj.generate_figures(P);
            
        end
        
        function [P,S,optAssignment,from_unique,cSeq,C] = find_optimal_context_labels(obj,S)
            
            iResamp = obj.resample_indices(S);
            
            cSeq = obj.context_sequences(S,iResamp);

            [C,~,~,P.mode_number_of_contexts] = obj.posterior_number_of_contexts(cSeq,S);
            
            % context label permutations
            L = flipud(perms(1:max(P.mode_number_of_contexts)));
            L = permute(L,[2,3,1]);
            nPerm = factorial(max(P.mode_number_of_contexts));
            
            % number of trials
            T = numel(obj.perturbations);
            
            % preallocate memory
            f = cell(1,T);
            to_unique = cell(1,T);
            from_unique = cell(1,T);
            optAssignment = cell(1,T);
            for trial = 1:T

                if ~mod(trial,50)
                    fprintf('finding the typical context sequence and the optimal context labels, trial = %d\n',trial)
                end

                    % exclude sequences for which C > max(P.mode_number_of_contexts) as 
                    % these sequences (and their descendents) will never be 
                    % analysed
                    f{trial} = find(C(:,trial) <= max(P.mode_number_of_contexts));

                    % identify unique sequences (to avoid performing the
                    % same computations multiple times)
                    [uniqueSequences,to_unique{trial},from_unique{trial}] = unique(cSeq{trial}(f{trial},:),'rows');

                    % number of unique sequences
                    nSeq = size(uniqueSequences,1);

                    % identify particles that have the same number of
                    % contexts as the most common number of contexts (only
                    % these particles will be analysed)
                    idx = logical(C(f{trial},trial) == P.mode_number_of_contexts(trial));

                    if trial == 1                       
                        % Hamming distances on trial 1
                        % dimension 3 of H considers all possible label permutations
                        H = double(1~=L(1,:,:));
                    elseif trial > 1
                        % identify a valid parent of each unique sequence
                        % (i.e. a sequence on the previous trial that is identical 
                        % up to the previous trial)
                        [i,~] = find(f{trial-1} == iResamp(f{trial}(to_unique{trial}),trial)');
                        parent = from_unique{trial-1}(i);

                        % pass Hamming distances from parents to children
                        i = sub2ind(size(H),repmat(parent,[1,nSeq,nPerm]),repmat(parent',[nSeq,1,nPerm]),repmat(permute(1:nPerm,[1,3,2]),[nSeq,nSeq,1]));
                        H = H(i);
                        
                        % recursively update Hamming distances
                        % dimension 3 of H considers all possible label permutations
                        for seq = 1:nSeq
                            H(seq:end,seq,:) = H(seq:end,seq,:) + double(uniqueSequences(seq,end) ~= L(uniqueSequences(seq:end,end),:,:));
                            H(seq,seq:end,:) = H(seq:end,seq,:); % Hamming distance is symmetric
                        end
                    end

                    % compute the Hamming distance between each pair of
                    % sequences (after optimally permuting labels)
                    Hopt = min(H,[],3);
                    
                    % count the number of times each unique sequence occurs
                    seqCnt = sum(from_unique{trial}(idx) == 1:size(uniqueSequences,1),1);
                    
                    if numel(unique(S.weights)) > 1
                        warning('the typical context sequence does not consider the nonuniform nature of run weights.')
                    end

                    % compute the mean optimal Hamming distance of each 
                    % sequence to all other sequences. the distance from
                    % sequence i to sequence j is weighted by the number of
                    % times sequence j occurs. if i == j, this weight is 
                    % reduced by 1 so that the distance from one instance 
                    % of sequence i to itself is ignored
                    Hmean = mean(Hopt.*(seqCnt-eye(nSeq)),2);
                    
                    % assign infinite distance to invalid sequences (i.e.
                    % sequences for which the number of contexts is not equal 
                    % to the most common number of contexts)
                    Hmean(seqCnt == 0) = Inf;

                    % find the index of the typical sequence (the sequence
                    % with the minimum mean optimal Hamming distance to all
                    % other sequences)
                    [~,i] = min(Hmean);

                    % typical sequence
                    typicalSequence = uniqueSequences(i,:);

                    % store the optimal permutation of labels for each sequence 
                    % with respect to the typical sequence
                    [~,j] = min(H(i,:,:),[],3);
                    optAssignment{trial} = permute(L(1:P.mode_number_of_contexts(trial),:,j),[3,1,2]);

            end

        end
        
        function iResamp = resample_indices(obj,S)
            
            if ~isfield(S.runs{1},'iResamp')
                error('The plot_COIN method cannot be used as no plot flags were activated in properties prior to calling the simulate_COIN method.')
            end
            
            % number of trials
            T = numel(obj.perturbations);
            
            iResamp = zeros(obj.particles*obj.runs,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                iResamp(p,:) = obj.particles*(run-1) + S.runs{run}.iResamp;
            end
            
        end
        
        function cSeq = context_sequences(obj,S,iResamp)
            
            % number of trials
            T = numel(obj.perturbations);
            
            cSeq = cell(1,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                for trial = 1:T
                    if run == 1
                        cSeq{trial} = zeros(obj.particles*obj.runs,trial);
                    end
                    if trial > 1
                        cSeq{trial}(p,1:trial-1) = cSeq{trial-1}(p,:);
                        cSeq{trial}(p,:) = cSeq{trial}(iResamp(p,trial),:);
                    end
                    cSeq{trial}(p,trial) = S.runs{run}.c(:,trial);
                end
            end
            
        end
        
        function [C,posterior,posteriorMean,posteriorMode] = posterior_number_of_contexts(obj,cSeq,S)
            
            % number of trials
            T = numel(obj.perturbations);
            
            % number of contexts
            C = zeros(obj.particles*obj.runs,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                for trial = 1:T
                    C(p,trial) = max(cSeq{trial}(p,:),[],2);
                end
            end

            particleWeight = repelem(S.weights,obj.particles)/obj.particles;
            if isrow(particleWeight)
                particleWeight = particleWeight';
            end

            posterior = zeros(obj.max_contexts+1,T);
            posteriorMean = zeros(1,T);
            posteriorMode = zeros(1,T);
            for time = 1:T

                for context = 1:max(C(:,time),[],'all')
                    posterior(context,time) = sum((C(:,time) == context).*particleWeight);
                end
                posteriorMean(time) = (1:obj.max_contexts+1)*posterior(:,time);
                [~,posteriorMode(time)] = max(posterior(:,time));

            end
            
        end
        
        function [P,S] = compute_variables_for_plotting(obj,P,S,optAssignment,from_unique,cSeq,C)

            % number of trials
            T = numel(obj.perturbations);
            
            P = obj.preallocate_memory_for_plot_variables(P);
            
            nParticlesUsed = zeros(T,obj.runs);
            for trial = 1:T
                if ~mod(trial,50)
                    fprintf('permuting the context labels, trial = %d\n',trial)
                end
                
                % cumulative number of particles for which C <= max(P.mode_number_of_contexts)
                N = 0;

                for run = 1:obj.runs
                    % indices of particles of the current run
                    p = obj.particles*(run-1) + (1:obj.particles);
                    
                    % indices of particles that are either valid now or 
                    % could be valid in the future: C <= max(P.mode_number_of_contexts)
                    validFuture = find(C(p,trial) <= max(P.mode_number_of_contexts));
                    
                    % indices of particles that are valid now: C == P.mode_number_of_contexts(trial)
                    validNow = find(C(p,trial) == P.mode_number_of_contexts(trial))';
                    nParticlesUsed(trial,run) = numel(validNow);
                    
                    if ~isempty(validNow)
                        for particle = validNow
                            % index of the optimal label permutations of the
                            % current particle
                            i = N + find(particle == validFuture);
                            
                            % is the latest context a novel context?
                            % this is needed to store novel context probabilities
                            cTraj = cSeq{trial}(p(particle),:);
                            novelContext = cTraj(trial) > max(cTraj(1:trial-1));
                            
                            S = obj.relabel_context_variables(S,optAssignment{trial}(from_unique{trial}(i),:),novelContext,particle,trial,run);
                        end
                        P = obj.integrate_over_particles(S,P,validNow,trial,run);
                    end
                    N = N + numel(validFuture);
                end
            end
            
            % (weighted) average number of particles used per run in analysis
            P.average_number_of_particles_used = obj.weighted_sum_along_dimension(nParticlesUsed,S,2);

            P = obj.integrate_over_runs(P,S);

            P = obj.normalise_relabelled_variables(P,nParticlesUsed,S);

            if obj.plot_state_given_context
                % the predicted state distribution for a novel context is the marginal
                % stationary distribution of the state after integrating out 
                % the drift and retention parameters under the prior
                P.state_given_novel_context = repmat(nanmean(P.state_given_context(:,:,end),2),[1,T]);
                P.state_given_context = P.state_given_context(:,:,1:end-1);
            end

        end

        function S = relabel_context_variables(obj,S,optAssignment,novelContext,particle,trial,run)

            C = numel(optAssignment);
            
            % number of trials
            T = numel(obj.perturbations);

            % predictive distributions
            if trial < T
                if obj.plot_state_given_context
                    S.runs{run}.xPred(optAssignment,particle,trial+1) = S.runs{run}.xPred(1:C,particle,trial+1);
                    S.runs{run}.vPred(optAssignment,particle,trial+1) = S.runs{run}.vPred(1:C,particle,trial+1);
                end
                if obj.plot_predicted_probabilities
                    S.runs{run}.cPred(optAssignment,particle,trial+1) = S.runs{run}.cPred(1:C,particle,trial+1);
                end
            end

            if obj.plot_responsibilities
                if trial == 1 || novelContext 
                    optAssignment2 = [optAssignment(1:end-1) C+1 optAssignment(end)];
                else
                    optAssignment2 = [optAssignment C+1];
                end
                S.runs{run}.cFilt(1:C+1,particle,trial) = S.runs{run}.cFilt(optAssignment2,particle,trial);
            end
            if obj.plot_stationary_probabilities
                S.runs{run}.cInf(optAssignment,particle,trial) = S.runs{run}.cInf(1:C,particle,trial);
            end
            if obj.plot_retention_given_context || obj.plot_drift_given_context
                S.runs{run}.adMu(:,optAssignment,particle,trial) = S.runs{run}.adMu(:,1:C,particle,trial);
                S.runs{run}.adCovar(:,:,optAssignment,particle,trial) = S.runs{run}.adCovar(:,:,1:C,particle,trial);
            end
            if obj.plot_bias_given_context
                S.runs{run}.bMu(optAssignment,particle,trial) = S.runs{run}.bMu(1:C,particle,trial);
                S.runs{run}.bVar(optAssignment,particle,trial) = S.runs{run}.bVar(1:C,particle,trial);
            end
            if obj.plot_global_transition_probabilities
                S.runs{run}.betaPosterior(optAssignment,particle,trial) = S.runs{run}.betaPosterior(1:C,particle,trial);
            end
            if obj.plot_local_transition_probabilities
                S.runs{run}.transitionMatrix(1:C,1:C+1,particle,trial) = ...
                obj.permute_transition_matrix_columns_and_rows(S.runs{run}.transitionMatrix(1:C,1:C+1,particle,trial),optAssignment);
            end
            if obj.plot_local_cue_probabilities
                S.runs{run}.emissionMatrix(optAssignment,:,particle,trial) = S.runs{run}.emissionMatrix(1:C,:,particle,trial);
            end

        end
        
        function P = permute_transition_matrix_columns_and_rows(obj,T,optAssignment)

                C = numel(optAssignment);

                [i_map,~] = find(optAssignment' == 1:C); % inverse mapping

                i = sub2ind(size(T),repmat(i_map,[1,C+1]),repmat([i_map' C+1],[C,1]));
                P = T(i);

        end
        
        function P = preallocate_memory_for_plot_variables(obj,P)
            
            % number of trials
            T = numel(obj.perturbations);

            if obj.plot_state_given_context
                P.state_given_context = NaN(numel(obj.state_values),T,max(P.mode_number_of_contexts)+1,obj.runs);
            end
            if obj.plot_predicted_probabilities
                P.predicted_probabilities = NaN(T,max(P.mode_number_of_contexts)+1,obj.runs);
                P.predicted_probabilities(1,end,:) = 1;
            end
            if obj.plot_state_given_context && obj.plot_predicted_probabilities
                P.xHatReduced = NaN(T,obj.runs);
            end
            if obj.plot_responsibilities
                P.responsibilities = NaN(T,max(P.mode_number_of_contexts)+1,obj.runs);
            end
            if obj.plot_stationary_probabilities
                P.stationary_probabilities = NaN(T,max(P.mode_number_of_contexts)+1,obj.runs);
            end
            if obj.plot_retention_given_context 
                P.retention_given_context = zeros(numel(obj.retention_values),T,max(P.mode_number_of_contexts),obj.runs);
            end
            if obj.plot_drift_given_context
                P.drift_given_context = zeros(numel(obj.drift_values),T,max(P.mode_number_of_contexts),obj.runs);
            end
            if obj.plot_bias_given_context
                P.bias_given_context = zeros(numel(obj.bias_values),T,max(P.mode_number_of_contexts),obj.runs);
            end
            if obj.plot_global_transition_probabilities
                P.global_transition_probabilities = NaN(T,max(P.mode_number_of_contexts)+1,obj.runs);
            end
            if obj.plot_local_transition_probabilities
                P.local_transition_probabilities = NaN(max(P.mode_number_of_contexts),max(P.mode_number_of_contexts)+1,T,obj.runs);
            end
            if obj.plot_global_cue_probabilities
                P.global_cue_probabilities = NaN(T,max(obj.cues)+1,obj.runs);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities = NaN(max(P.mode_number_of_contexts),max(obj.cues)+1,T,obj.runs);
            end
            if obj.plot_state
                P.state = NaN(numel(obj.state_values),T,obj.runs);
            end
            if obj.plot_average_state || obj.plot_state
                P.average_state = NaN(T,obj.runs);
            end
            if obj.plot_bias
                P.bias = NaN(numel(obj.bias_values),T,obj.runs);
            end
            if obj.plot_average_bias || obj.plot_bias
                P.average_bias = NaN(T,obj.runs);
            end
            if obj.plot_state_feedback
                P.state_feedback = NaN(numel(obj.state_feedback_values),T,obj.runs);
            end
            if obj.plot_explicit_component
                P.explicit_component = NaN(T,obj.runs);
            end
            if obj.plot_implicit_component
                P.implicit_component = NaN(T,obj.runs);
            end
            P.average_state_feedback = NaN(T,obj.runs);  

        end
        
        function P = integrate_over_particles(obj,S,P,particles,trial,run)

            C = P.mode_number_of_contexts(trial);
            novelContext = max(P.mode_number_of_contexts)+1;
            
            % number of trials
            T = numel(obj.perturbations);

            % predictive distributions
            if trial < T
                if obj.plot_state_given_context
                    mu = permute(S.runs{run}.xPred(1:C+1,particles,trial+1),[3,2,1]);
                    sd = permute(sqrt(S.runs{run}.vPred(1:C+1,particles,trial+1)),[3,2,1]);
                    P.state_given_context(:,trial+1,[1:C novelContext],run) = sum(normpdf(obj.state_values',mu,sd),2);
                end
                if obj.plot_predicted_probabilities
                    P.predicted_probabilities(trial+1,[1:C novelContext],run) = sum(S.runs{run}.cPred(1:C+1,particles,trial+1),2);
                end
                if obj.plot_state_given_context && obj.plot_predicted_probabilities
                    P.xHatReduced(trial+1,run) = sum(S.runs{run}.cPred(:,particles,trial+1).*S.runs{run}.xPred(:,particles,trial+1),'all');
                end
            end

            if obj.plot_responsibilities
                P.responsibilities(trial,[1:C novelContext],run) = sum(S.runs{run}.cFilt(1:C+1,particles,trial),2);
            end
            if obj.plot_stationary_probabilities
                P.stationary_probabilities(trial,[1:C novelContext],run) = sum(S.runs{run}.cInf(1:C+1,particles,trial),2);
            end
            if obj.plot_retention_given_context 
                mu = permute(S.runs{run}.adMu(1,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S.runs{run}.adCovar(1,1,1:C,particles,trial)),[1,4,3,2]);
                P.retention_given_context(:,trial,1:C,run) = sum(normpdf(obj.retention_values',mu,sd),2);
            end
            if obj.plot_drift_given_context
                mu = permute(S.runs{run}.adMu(2,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S.runs{run}.adCovar(2,2,1:C,particles,trial)),[1,4,3,2]);
                P.drift_given_context(:,trial,1:C,run) = sum(normpdf(obj.drift_values',mu,sd),2);  
            end
            if obj.plot_bias_given_context
                mu = permute(S.runs{run}.bMu(1:C,particles,trial),[3,2,1]);
                sd = permute(sqrt(S.runs{run}.bVar(1:C,particles,trial)),[3,2,1]);
                P.bias_given_context(:,trial,1:C,run) = sum(normpdf(obj.bias_values',mu,sd),2);  
            end
            if obj.plot_global_transition_probabilities
                alpha = S.runs{run}.betaPosterior(1:C+1,particles,trial);
                P.global_transition_probabilities(trial,[1:C novelContext],run) = sum(alpha./sum(alpha,1),2);
            end
            if obj.plot_local_transition_probabilities
                P.local_transition_probabilities(1:C,[1:C novelContext],trial,run) = sum(S.runs{run}.transitionMatrix(1:C,1:C+1,particles,trial),3);
            end
            if obj.plot_global_cue_probabilities
                alpha = S.runs{run}.betaEPosterior(1:max(obj.cues(1:trial))+1,:,trial);
                P.global_cue_probabilities(trial,[1:max(obj.cues(1:trial)) max(obj.cues)+1],run) = sum(alpha./sum(alpha,1),2);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities(1:C,[1:max(obj.cues(1:trial)) max(obj.cues)+1],trial,run) = sum(S.runs{run}.emissionMatrix(1:C,1:max(obj.cues(1:trial))+1,particles,trial),3);
            end

        end
        
        function P = integrate_over_runs(obj,P,S)

            if obj.plot_state_given_context
                P.state_given_context = obj.weighted_sum_along_dimension(P.state_given_context,S,4);
            end
            if obj.plot_predicted_probabilities
                P.predicted_probabilities = obj.weighted_sum_along_dimension(P.predicted_probabilities,S,3);
            end
            if obj.plot_state_given_context && obj.plot_predicted_probabilities
                P.xHatReduced = obj.weighted_sum_along_dimension(P.xHatReduced,S,2);
            end
            if obj.plot_responsibilities
                P.responsibilities = obj.weighted_sum_along_dimension(P.responsibilities,S,3);
            end
            if obj.plot_stationary_probabilities
                P.stationary_probabilities = obj.weighted_sum_along_dimension(P.stationary_probabilities,S,3);
            end
            if obj.plot_retention_given_context 
                P.retention_given_context = obj.weighted_sum_along_dimension(P.retention_given_context,S,4);
            end
            if obj.plot_drift_given_context
                P.drift_given_context = obj.weighted_sum_along_dimension(P.drift_given_context,S,4);
            end
            if obj.plot_bias_given_context
                P.bias_given_context = obj.weighted_sum_along_dimension(P.bias_given_context,S,4);
            end
            if obj.plot_global_transition_probabilities
                P.global_transition_probabilities = obj.weighted_sum_along_dimension(P.global_transition_probabilities,S,3);
            end
            if obj.plot_local_transition_probabilities
                P.local_transition_probabilities = obj.weighted_sum_along_dimension(P.local_transition_probabilities,S,4);
            end
            if obj.plot_global_cue_probabilities
                P.global_cue_probabilities = obj.weighted_sum_along_dimension(P.global_cue_probabilities,S,3);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities = obj.weighted_sum_along_dimension(P.local_cue_probabilities,S,4);
            end
            
            if obj.plot_state
                for run = 1:obj.runs
                    P.state(:,:,run) = S.runs{run}.xPredMarg;
                end
                P.state = obj.weighted_sum_along_dimension(P.state,S,3);
            end
            if obj.plot_average_state || obj.plot_state
                for run = 1:obj.runs
                    P.average_state(:,run) = S.runs{run}.xHat;
                end
                P.average_state = obj.weighted_sum_along_dimension(P.average_state,S,2);
            end
            if obj.plot_bias
                for run = 1:obj.runs
                    P.bias(:,:,run) = S.runs{run}.bPredMarg;
                end
                P.bias = obj.weighted_sum_along_dimension(P.bias,S,3);
            end
            if obj.plot_average_bias || obj.plot_bias
                for run = 1:obj.runs
                    P.average_bias(:,run) = S.runs{run}.implicit;
                end
                P.average_bias = obj.weighted_sum_along_dimension(P.average_bias,S,2);
            end
            if obj.plot_state_feedback
                for run = 1:obj.runs
                    P.state_feedback(:,:,run) = S.runs{run}.yPredMarg;
                end
                P.state_feedback = obj.weighted_sum_along_dimension(P.state_feedback,S,3);
            end
            if obj.plot_explicit_component
                for run = 1:obj.runs
                    P.explicit_component(:,run) = S.runs{run}.explicit;
                end
                P.explicit_component = obj.weighted_sum_along_dimension(P.explicit_component,S,2);
            end
            if obj.plot_implicit_component
                for run = 1:obj.runs
                    P.implicit_component(:,run) = S.runs{run}.implicit;
                end
                P.implicit_component = obj.weighted_sum_along_dimension(P.implicit_component,S,2);
            end
            for run = 1:obj.runs
                P.average_state_feedback(:,run) = S.runs{run}.yHat;
            end
            P.average_state_feedback = obj.weighted_sum_along_dimension(P.average_state_feedback,S,2);

        end
        
        function X = weighted_sum_along_dimension(obj,X,S,dim)

            % find elements that are NaN throughout dimension dim
            i = all(isnan(X),dim);

            % sum over dimension dim of X with weights w
            X = sum(X.*reshape(S.weights,[ones(1,dim-1) numel(S.weights)]),dim,'omitnan');

            % elements that are NaN throughout dimension dim should remain NaN, not 0
            X(i) = NaN;

        end
        
        function P = normalise_relabelled_variables(obj,P,nParticlesUsed,S)
            
            % number of trials
            T = numel(obj.perturbations);

            % normalisation constant
            Z = sum(nParticlesUsed.*S.weights,2);

            if obj.plot_state_given_context
                P.state_given_context(:,2:end,:) = P.state_given_context(:,2:end,:)./Z(1:end-1)';
            end
            if obj.plot_predicted_probabilities
                P.predicted_probabilities(2:end,:) = P.predicted_probabilities(2:end,:)./Z(1:end-1);
            end
            if obj.plot_state_given_context && obj.plot_predicted_probabilities
                P.xHatReduced(2:end,:) = P.xHatReduced(2:end,:)./Z(1:end-1);
            end
            if obj.plot_responsibilities
                P.responsibilities = P.responsibilities./Z;
                P.novel_context_probability = P.responsibilities(:,end);
                P.responsibilities = P.responsibilities(:,1:end-1);
            end
            if obj.plot_stationary_probabilities
                P.stationary_probabilities = P.stationary_probabilities./Z;
            end
            if obj.plot_retention_given_context 
                P.retention_given_context = P.retention_given_context./Z';
            end
            if obj.plot_drift_given_context
                P.drift_given_context = P.drift_given_context./Z';
            end
            if obj.plot_bias_given_context
                P.bias_given_context = P.bias_given_context./Z';
            end
            if obj.plot_global_transition_probabilities
                P.global_transition_probabilities = P.global_transition_probabilities./Z;
            end
            if obj.plot_local_transition_probabilities
                P.local_transition_probabilities = P.local_transition_probabilities./reshape(Z,[1,1,T]);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities = P.local_cue_probabilities./reshape(Z,[1,1,T]);
            end

        end
        
        function plot_image(obj,C,YLims,YTicks,RGB)

            data = zeros([size(C,[1,2]) 3]);
            for context = 1:size(C,3)
                intensity = C(:,:,context)/max(C(:,:,context),[],'all');
                for rgb = 1:3
                    data(:,:,rgb) = nansum(cat(3,data(:,:,rgb),(1-RGB(context,rgb)).*intensity),3);
                end
            end
            if issorted(YLims)
                data = flipud(data);
            end
            data = 1-data;

            imagesc(data);
            nPixels = size(C,1);
            YTicksPixels = obj.map_to_pixel_space(nPixels,YLims,YTicks);
            YTicksPixels = sort(YTicksPixels,'ascend');
            YTicks = sort(YTicks,'descend');
            set(gca,'YTick',YTicksPixels,'YTickLabels',YTicks)

        end
        
        function YTicksPixels = map_to_pixel_space(obj,nPixels,Lims,YTicks)

            % map points to pixel space

            % imagesc plots pixels of size 1 centered on the integers (e.g., the first 
            % pixel is centered on 1 and spans from 0.5 to 1.5)

            Lims = sort(Lims,'descend');
            YTicksPixels = 1 + (nPixels-1)*((YTicks-Lims(1))/(Lims(2)-Lims(1)));

        end
        
        function generate_figures(obj,P)

            C = obj.colours;
            
            line_width = 2;
            font_size = 15;
            
            % number of trials
            T = numel(obj.perturbations);
            
            if obj.plot_state_given_context
                figure
                YLims = obj.state_values([1 end]);
                YTicks = [0 obj.state_values([1 end])];
                obj.plot_image(P.state_given_context,YLims,YTicks,C.contexts(1:max(P.mode_number_of_contexts),:))
                set(gca,'FontSize',font_size)
                ylabel('state | context')
                figure
                YLims = obj.state_values([1 end]);
                YTicks = [0 obj.state_values([1 end])];
                obj.plot_image(P.state_given_novel_context,YLims,YTicks,C.new_context)
                set(gca,'FontSize',font_size)
                ylabel('state | novel context')
            end
            if obj.plot_predicted_probabilities
                figure
                hold on
                plot(P.predicted_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(P.mode_number_of_contexts)
                    c = P.predicted_probabilities(:,context);
                    t = find(~isnan(P.predicted_probabilities(:,context)),1,'first');
                    c(t-1) = P.predicted_probabilities(t-1,end);
                    plot(c,'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('predicted probabilities')
            end
            if obj.plot_responsibilities
                figure
                hold on
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.responsibilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('responsibilities')
                figure
                plot(P.novel_context_probability,'Color',C.new_context,'LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('novel context probability')
            end
            if obj.plot_stationary_probabilities
                figure
                hold on
                plot(P.stationary_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.stationary_probabilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('stationary context probabilities')
            end
            if obj.plot_retention_given_context 
                figure
                YLims = obj.retention_values([1 end]);
                YTicks = [0 obj.retention_values([1 end])];
                obj.plot_image(P.retention_given_context,YLims,YTicks,C.contexts)
                set(gca,'FontSize',font_size)
                ylabel('retention | context')
            end
            if obj.plot_drift_given_context
                figure
                YLims = obj.drift_values([1 end]);
                YTicks = [0 obj.drift_values([1 end])];
                obj.plot_image(P.drift_given_context,YLims,YTicks,C.contexts)
                set(gca,'FontSize',font_size)
                ylabel('drift | context')
            end
            if obj.plot_bias_given_context
                figure
                YLims = obj.bias_values([1 end]);
                YTicks = [0 obj.bias_values([1 end])];
                obj.plot_image(P.bias_given_context,YLims,YTicks,C.contexts)
                set(gca,'FontSize',font_size)
                ylabel('bias | context')
            end
            if obj.plot_global_transition_probabilities
                figure
                hold on
                plot(P.global_transition_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.global_transition_probabilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('global transition probabilities')
            end
            if obj.plot_local_transition_probabilities
                for from_context = 1:max(P.mode_number_of_contexts)
                    figure
                    hold on
                    plot(squeeze(P.local_transition_probabilities(from_context,end,:)),'Color',C.new_context,'LineWidth',line_width)
                    for to_context = 1:max(P.mode_number_of_contexts)
                        plot(squeeze(P.local_transition_probabilities(from_context,to_context,:)),'Color',C.contexts(to_context,:),'LineWidth',line_width)
                    end
                    title(sprintf('\\color[rgb]{%s}context %d',num2str(C.contexts(from_context,:)),from_context))
                    axis([0 360 -0.1 1.1])
                    set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                    ylabel('local transition probabilities')
                end
            end
            if obj.plot_global_cue_probabilities
                figure
                hold on
                plot(P.global_cue_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(obj.cues)
                    plot(P.global_cue_probabilities(:,context),'Color',C.cues(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                ylabel('global cue probabilities')
            end
            if obj.plot_local_cue_probabilities
                for context = 1:max(P.mode_number_of_contexts)
                    figure
                    hold on
                    tmp = {};
                    plot(squeeze(P.local_cue_probabilities(context,end,:)),'Color',C.new_context,'LineWidth',line_width)
                    tmp = cat(2,tmp,'novel cue');
                    for cue = 1:max(obj.cues)
                        plot(squeeze(P.local_cue_probabilities(context,cue,:)),'Color',C.cues(cue,:),'LineWidth',line_width)
                        tmp = cat(2,tmp,sprintf('cue %d',cue));
                    end
                    title(sprintf('\\color[rgb]{%s}context %d',num2str(C.contexts(context,:)),context))
                    axis([0 360 -0.1 1.1])
                    set(gca,'YTick',[0 0.5 1],'FontSize',font_size)
                    ylabel('local cue probabilities')
                    legend(tmp,'location','best','box','off')
                end
            end
            if obj.plot_state
                figure
                YLims = obj.state_values([1 end]);
                YTicks = [0 obj.state_values([1 end])];
                obj.plot_image(P.state,YLims,YTicks,C.marginal)
                hold on
                nPixels = numel(obj.state_values);
                plot(obj.map_to_pixel_space(nPixels,YLims,P.average_state),'Color',C.mean_of_marginal,'LineWidth',line_width)
                set(gca,'FontSize',font_size)
                ylabel('state')
            end
            if obj.plot_average_state
                figure
                plot(P.average_state,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'FontSize',font_size)
                ylabel('average state')
            end
            if obj.plot_bias
                figure
                YLims = obj.bias_values([1 end]);
                YTicks = [0 obj.bias_values([1 end])];
                obj.plot_image(P.bias,YLims,YTicks,C.marginal)
                hold on
                nPixels = numel(obj.bias_values);
                plot(obj.map_to_pixel_space(nPixels,YLims,P.average_bias),'Color',C.mean_of_marginal,'LineWidth',line_width)  
                set(gca,'FontSize',font_size)
                ylabel('bias')
            end
            if obj.plot_average_bias
                figure
                plot(P.average_bias,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'FontSize',font_size)
                ylabel('average bias')
            end
            if obj.plot_state_feedback
                figure
                YLims = obj.state_feedback_values([1 end]);
                YTicks = [0 obj.state_feedback_values([1 end])];
                obj.plot_image(P.state_feedback,YLims,YTicks,C.marginal)
                hold on
                nPixels = numel(obj.state_feedback_values);
                plot(obj.map_to_pixel_space(nPixels,YLims,P.average_state_feedback ),'Color',C.mean_of_marginal,'LineWidth',line_width)  
                set(gca,'FontSize',font_size)
                ylabel('state feedback')
            end
            if obj.plot_explicit_component
                figure
                plot(P.explicit_component,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'FontSize',font_size)
                ylabel('adaptation (explicit component)')
            end
            if obj.plot_implicit_component
                figure
                plot(P.implicit_component,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'FontSize',font_size)
                ylabel('adaptation (implicit component)')
            end

        end
        
        function C = colours(obj)

            C.contexts = [0.1216    0.4706    0.7059
                          0.8902    0.1020    0.1098
                          1.0000    0.4980         0
                          0.2000    0.6275    0.1725
                          0.9843    0.6039    0.6000
                          0.8902    0.1020    0.1098
                          0.9922    0.7490    0.4353
                          1.0000    0.4980         0];
            C.new_context = 0.7*[1 1 1];
            C.marginal = [208 149 213]/255;
            C.mean_of_marginal = [54 204 255]/255;
            if ~isempty(obj.cues) 
                C.cues = cool(max(obj.cues));
            end
            
        end
        
    end
end