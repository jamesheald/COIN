classdef COIN < matlab.mixin.Copyable
    % COIN v1.0.0
    % Author: James Heald
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
    %         rho_context                          rho (normalised self-transition) hyperparameter of the Chinese restaurant franchise for the context transitions
    %     parameters if cues are present
    %         gamma_cue                            gamma hyperparameter of the Chinese restaurant franchise for the cue emissions
    %         alpha_cue                            alpha hyperparameter of the Chinese restaurant franchise for the cue emissions
    %     parameters if inferring bias
    %         infer_bias                           infer the measurment bias (true) or not (false)
    %         prior_precision_bias                 precision (inverse variance) of prior of measurement bias
    %     paradigm
    %         perturbations                        vector of perturbations (use NaN on channel trials)
    %         cues                                 vector of sensory cues (encode cues as consecutive integers starting from 1)
    %         stationary_trials                    trials on which to set the predicted probabilities to the stationary probabilities (e.g. following a working memory task)
    %     runs
    %         runs                                 number of runs, each conditioned on a different state feedback sequence
    %     parallel processing of runs
    %         max_cores                            maximum number of CPU cores available (0 implements serial processing of runs)
    %     model implementation
    %         particles                            number of particles
    %         max_contexts                         maximum number of contexts that can be instantiated
    %     measured adaptation data
    %         adaptation                           vector of adaptation data (use NaN on trials where adaptation was not measured)
    %     store
    %         store                                variables to store in memory
    %     plot flags
    %         plot_state_given_context             plot state | context distribution ('predicted state distribution for each context')
    %         plot_predicted_probabilities         plot predicted probabilities
    %         plot_responsibilities                plot responsibilities
    %         plot_stationary_probabilities        plot stationary probabilities
    %         plot_retention_given_context         plot retention | context distribution
    %         plot_drift_given_context             plot drift | context distribution
    %         plot_bias_given_context              plot bias | context distribution
    %         plot_global_transition_probabilities plot global transition probabilities
    %         plot_local_transition_probabilities  plot local transition probabilities
    %         plot_global_cue_probabilities        plot global cue probabilities
    %         plot_local_cue_probabilities         plot local cue probabilities
    %         plot_state                           plot state ('overall predicted state distribution')
    %         plot_average_state                   plot average state (mean of 'overall predicted state distribution')
    %         plot_bias                            plot bias distribution (average bias distribution across contexts)
    %         plot_average_bias                    plot average bias (mean of average bias distribution across contexts)
    %         plot_state_feedback                  plot predicted state feedback distribution (average state feedback distribution across contexts)
    %         plot_explicit_component              plot explicit component of learning
    %         plot_implicit_component              plot implicit component of learning
    %         plot_Kalman_gain_given_cstar1        plot Kalman gain | context with highest responsibility on current trial (cstar1)
    %         plot_predicted_probability_cstar1    plot predicted probability of context with highest responsibility on current trial (cstar1)
    %         plot_state_given_cstar1              plot state | context with highest responsibility on current trial (cstar1)
    %         plot_Kalman_gain_given_cstar2        plot Kalman gain | context with highest predicted probability on next trial (cstar2)
    %         plot_state_given_cstar2              plot state | context with highest predicted probability on next trial (cstar2)
    %         plot_predicted_probability_cstar3    plot predicted probability of context with highest predicted probability on current trial (cstar3)
    %         plot_state_given_cstar3              plot state | context with highest predicted probability on current trial (cstar3)
    %     plot inputs
    %         retention_values                     specify values at which to evaluate p(retention) if plot_retention_given_context == true
    %         drift_values                         specify values at which to evaluate p(drift) if plot_drift_given_context == true
    %         state_values                         specify values at which to evaluate p(state) if plot_state_given_context == true or plot_state == true
    %         bias_values                          specify values at which to evaluate p(bias) if plot_bias_given_context == true or plot_bias == true
    %         state_feedback_values                specify values at which to evaluate p(state feedback) if plot_state_feedback == true
    %     miscellaneous user data
    %         user_data                            any data the user would like to associate with an object of the class
    %
    % VARIABLES                                    description
    %         average_state                        average predicted state (average across contexts and particles)
    %         bias                                 bias of each context (sample)                   
    %         bias_distribution                    bias distribution (discretised)
    %         bias_mean                            mean of the posterior of the bias for each context
    %         bias_ss_1                            sufficient statistic #1 for the bias parameter of each context
    %         bias_ss_2                            sufficient statistic #2 for the bias parameter of each context
    %         bias_var                             variance of the posterior of the bias for each context
    %         C                                    number of instantiated contexts
    %         context                              context (sample)
    %         drift                                state drift of each context (sample)
    %         dynamics_covar                       covariance of the posterior of the retention and drift of each context
    %         dynamics_mean                        mean of the posterior of the retention and drift of each context
    %         dynamics_ss_1                        sufficient statistic #1 for the retention and drift parameters of each context
    %         dynamics_ss_2                        sufficient statistic #2 for the retention and drift parameters of each context
    %         explicit                             explicit component of learning
    %         global_cue_posterior                 parameters of the posterior of the global cue distribution                                      
    %         global_cue_probabilities             global cue distribution (sample)
    %         global_transition_posterior          parameters of the posterior of the global transition distribution  
    %         global_transition_probabilities      global transition distribution (sample)
    %         i_observed                           indices of observed states
    %         i_resampled                          indices of resampled particles
    %         implicit                             implicit component of learning
    %         Kalman_gains                         Kalman gain for each context
    %         local_cue_matrix                     expected local cue probability matrix
    %         local_transition_matrix              expected local context transition probability matrix
    %         m_context                            number of tables in restaurant i serving dish j (Chinese restaurant franchise for the context transitions)
    %         m_cue                                number of tables in restaurant i serving dish j (Chinese restaurant franchise for the cue emissions)
    %         motor_noise                          motor noise
    %         motor_output                         average predicted state feedback (average across contexts and particles) a.k.a the motor output
    %         n_context                            local context transition counts
    %         n_cue                                local cue emission counts
    %         predicted_probabilities              predicted context probabilities (conditioned on the cue)
    %         prediction_error                     state feedback prediction error for each context
    %         previous_context                     context sampled on the previous trial
    %         previous_state_filtered_mean         mean of the filtered state distribution for each context on the previous trial
    %         previous_state_filtered_var          variance of the filtered state distribution for each context on the previous trial
    %         previous_x_dynamics                  samples of the states on the previous trial (to update the sufficient statistics for the retention and drift parameters of each context)
    %         prior_probabilities                  prior context probabilities (not conditioned on the cue)
    %         probability_cue                      probability of the observed cue for each context        
    %         probability_state_feedback           probability of the observed state feedback for each context
    %         Q                                    number of cues observed
    %         responsibilities                     context responsibilities (conditioned on the cue and the state feedback)
    %         retention                            state retention factor of each context (sample)
    %         sensory_noise                        sensory noise
    %         state_distribution                   predicted state distribution (discretised)
    %         state_feedback_distribution          predicted state feedback distribution (discretised)
    %         state_feedback_mean                  mean of the predicted state feedback distribution for each context
    %         state_feedback_var                   variance of the predicted state feedback distribution for each context
    %         state_filtered_mean                  mean of the filtered state distribution for each context 
    %         state_filtered_var                   variance of the filtered state distribution for each context 
    %         state_mean                           mean of the predicted state distribution for each context
    %         state_var                            variance of the predicted state distribution for each context
    %         stationary_probabilities             stationary context probabilities
    %         x_bias                               samples of the states on the current trial (to update the sufficient statistics for the bias parameter of each context)
    %         x_dynamics                           samples of the states on the current trial (to update the sufficient statistics for the retention and drift parameters of each context)
    
    properties
        
        % core parameters - values taken from Heald et al. (2020) Table S1 (A)
        sigma_process_noise = 0.0089 
        sigma_sensory_noise = 0.03
        sigma_motor_noise = 0.0182
        prior_mean_retention = 0.9425 
        prior_precision_retention = 837.1^2
        prior_precision_drift = 1.2227e+3^2
        gamma_context = 0.1
        alpha_context = 8.955
        rho_context = 0.2501
        
        % parameters if cues are present
        gamma_cue = 0.1
        alpha_cue = 25
        
        % parameters if inferring a bias
        infer_bias = false
        prior_precision_bias = 70^2
        
        % paradigm
        perturbations
        cues
        stationary_trials
        
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
        store = {'state_feedback','motor_output'}
        
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
        plot_Kalman_gain_given_cstar1 = false
        plot_predicted_probability_cstar1 = false
        plot_state_given_cstar1 = false
        plot_Kalman_gain_given_cstar2 = false
        plot_state_given_cstar2 = false
        plot_predicted_probability_cstar3 = false
        plot_state_given_cstar3 = false
        
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
            
            if ~isempty(obj.cues) 
                obj.check_cue_labels;
            end
            
            % set the store property based on the plots requested
            obj_store = copy(obj);
            obj_store = set_store_property_for_plots(obj_store);
            
            % number of trials
            T = numel(obj.perturbations);
            
            % preallocate memory
            tmp = cell(1,obj.runs);
            
            if isempty(obj.adaptation)
                
                trials = 1:T;
                
                % perform runs
                fprintf('Simulating the COIN model.\n')
                parfor (run = 1:obj.runs,obj.max_cores)
                    tmp{run} = obj_store.main_loop(trials).stored;
                end
                
                % assign equal weights to all runs
                w = ones(1,obj.runs)/obj.runs;

            else
                
                if numel(obj.adaptation) ~= numel(obj.perturbations)
                    error('Property ''adaptation'' should be a vector with one element per trial (use NaN on trials where adaptation was not measured).')
                end
                
                % perform runs
                % resample runs whenever the effective sample size falls below threshold

                % preallocate memory
                D_in = cell(1,obj.runs);
                D_out = cell(1,obj.runs);
                
                % initialise weights to be uniform
                w = ones(1,obj.runs)/obj.runs;

                % effective sample size threshold for resampling
                ESS_threshold = 0.5*obj.runs;

                % trials on which adaptation was measured
                adaptation_trials = find(~isnan(obj.adaptation));

                % simulate trials inbetween trials on which adaptation was measured
                for i = 1:numel(adaptation_trials)

                    if i == 1
                        trials = 1:adaptation_trials(i);
                        fprintf('Simulating the COIN model from trial 1 to trial %d.\n',adaptation_trials(i))
                    else
                        trials = adaptation_trials(i-1)+1:adaptation_trials(i);
                        fprintf('Simulating the COIN model from trial %d to trial %d.\n',adaptation_trials(i-1)+1,adaptation_trials(i))
                    end

                    parfor (run = 1:obj.runs,obj.max_cores)
                        if i == 1
                            D_out{run} = obj_store.main_loop(trials);
                        else
                            D_out{run} = obj_store.main_loop(trials,D_in{run});
                        end
                    end

                    % calculate the log likelihood
                    log_likelihood = zeros(1,obj.runs);
                    for run = 1:obj.runs
                        model_error = D_out{run}.stored.motor_output(adaptation_trials(i)) - obj.adaptation(adaptation_trials(i));
                        log_likelihood(run) = -(log(2*pi*obj.sigma_motor_noise^2) + (model_error/obj.sigma_motor_noise).^2)/2; 
                    end

                    % update the weights and normalise
                    l_w = log_likelihood + log(w);
                    l_w = l_w - obj.log_sum_exp(l_w');
                    w = exp(l_w);

                    % calculate the effective sample size
                    ESS = 1/(sum(w.^2));

                    % if the effective sample size falls below ESS_threshold, resample
                    if ESS < ESS_threshold
                        fprintf('Effective sample size = %.1f %s resampling runs.\n',ESS,char(8212))
                        i_resampled = obj.systematic_resampling(w);
                        for run = 1:obj.runs
                            D_in{run} = D_out{i_resampled(run)};
                        end
                        w = ones(1,obj.runs)/obj.runs;
                    else
                        fprintf('Effective sample size = %.1f.\n',ESS)
                        D_in = D_out;
                    end

                end
                
                if adaptation_trials(end) == T
                    
                    for run = 1:obj.runs
                        tmp{run} = D_in{run}.stored;
                    end

                elseif adaptation_trials(end) < T

                    % simulate to the last trial
                    fprintf('Simulating the COIN model from trial %d to trial %d.\n',adaptation_trials(end)+1,T)

                    trials = adaptation_trials(end)+1:T;

                    parfor (run = 1:obj.runs,obj.max_cores)
                        tmp{run} = obj_store.main_loop(trials,D_in{run}).stored;
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
            
            % generate plots
            props = properties(obj);
            for i = find(contains(props','plot'))
                if obj.(props{i})
                    S.plots = plot_COIN(obj,S);
                    break
                end
            end
            
            % delete the raw variables that were stored to generate the plots
            field_names = fieldnames(S.runs{1});
            for i = 1:length(field_names)
                if all(~strcmp(field_names{i},obj.store))
                    for run = 1:obj.runs
                        S.runs{run} = rmfield(S.runs{run},field_names{i});
                    end
                end
            end
            
        end
        
        function obj = set_store_property_for_plots(obj)

            % specify variables that need to be stored for plots
            tmp = {};
            if obj.plot_state_given_context
                tmp = cat(2,tmp,{'state_mean','state_var'});
            end
            if obj.plot_predicted_probabilities
                tmp = cat(2,tmp,'predicted_probabilities');
            end
            if obj.plot_responsibilities
                tmp = cat(2,tmp,'responsibilities');
            end
            if obj.plot_stationary_probabilities
                tmp = cat(2,tmp,'stationary_probabilities');
            end
            if obj.plot_retention_given_context
                tmp = cat(2,tmp,{'dynamics_mean','dynamics_covar'});
            end
            if obj.plot_drift_given_context
                tmp = cat(2,tmp,{'dynamics_mean','dynamics_covar'});
            end
            if obj.plot_bias_given_context
                if obj.infer_bias
                    tmp = cat(2,tmp,{'bias_mean','bias_var'});
                else
                    error('You must infer the measurement bias parameter to use plot_bias_given_context. Set property ''infer_bias'' to true.')
                end
            end
            if obj.plot_global_transition_probabilities
                tmp = cat(2,tmp,'global_transition_posterior');
            end
            if obj.plot_local_transition_probabilities
                tmp = cat(2,tmp,'local_transition_matrix');
            end
            if obj.plot_local_cue_probabilities
                if isempty(obj.cues) 
                    error('An experiment must have sensory cues to use plot_local_cue_probabilities.')
                else
                    tmp = cat(2,tmp,'local_cue_matrix');
                end
            end
            if obj.plot_global_cue_probabilities
                if isempty(obj.cues) 
                    error('An experiment must have sensory cues to use plot_global_cue_probabilities.')
                else
                    tmp = cat(2,tmp,'global_cue_posterior');
                end
            end
            if obj.plot_state
                tmp = cat(2,tmp,'state_distribution','average_state');
            end
            if obj.plot_average_state
                tmp = cat(2,tmp,'average_state');
            end
            if obj.plot_bias
                if obj.infer_bias
                    tmp = cat(2,tmp,'bias_distribution','implicit');
                else
                    error('You must infer the measurement bias parameter to use plot_bias. Set property ''infer_bias'' to true.')
                end
            end
            if obj.plot_average_bias
                tmp = cat(2,tmp,'implicit');
            end
            if obj.plot_state_feedback
                tmp = cat(2,tmp,'state_feedback_distribution');
            end
            if obj.plot_explicit_component
                tmp = cat(2,tmp,'explicit');
            end
            if obj.plot_implicit_component
                tmp = cat(2,tmp,'implicit');
            end
            if obj.plot_Kalman_gain_given_cstar1
                tmp = cat(2,tmp,'Kalman_gain_given_cstar1');
            end
            if obj.plot_predicted_probability_cstar1
                tmp = cat(2,tmp,'predicted_probability_cstar1');
            end
            if obj.plot_state_given_cstar1
                tmp = cat(2,tmp,'state_given_cstar1');
            end
            if obj.plot_Kalman_gain_given_cstar2
                tmp = cat(2,tmp,'Kalman_gain_given_cstar2');
            end
            if obj.plot_state_given_cstar2
                tmp = cat(2,tmp,'state_given_cstar2');
            end 
            if obj.plot_predicted_probability_cstar3
                tmp = cat(2,tmp,'predicted_probability_cstar3');
            end
            if obj.plot_state_given_cstar3
                tmp = cat(2,tmp,'state_given_cstar3');
            end 
            if ~isempty(tmp)
                tmp = cat(2,tmp,{'context','i_resampled'});
            end
            
            % add strings in tmp to the store property of obj
            for i = 1:numel(tmp)
                if ~any(strcmp(obj.store,tmp{i}))
                    obj.store{end+1} = tmp{i};
                end
            end
            
        end
        
        function objective = objective_COIN(obj)
            
            P = numel(obj); % number of participants
            n = sum(~isnan(obj(1).adaptation)); % number of adaptation measurements per participant
            adaptation_trials = zeros(n,P);
            data = zeros(n,P);
            for p = 1:P
                
                if numel(obj(p).adaptation) ~= numel(obj(p).perturbations)
                    error('Property ''adaptation'' should be a vector with one element per trial (use NaN on trials where adaptation was not measured).')
                end

                if isrow(obj(p).adaptation)
                    obj(p).adaptation = obj(p).adaptation';
                end
                
                % trials on which adaptation was measured
                adaptation_trials(:,p) = find(~isnan(obj(p).adaptation));

                % measured adaptation
                data(:,p) = obj(p).adaptation(adaptation_trials(:,p));
                
            end

            log_likelihood = zeros(obj(1).runs,1);
            parfor (run = 1:obj(1).runs,obj(1).max_cores)

                model = zeros(n,P);
                for p = 1:P
                    
                    % number of trials
                    T = numel(obj(p).perturbations);
                    
                    trials = 1:T;

                    % model adaptation
                    model(:,p) = obj(p).main_loop(trials).stored.motor_output(adaptation_trials(:,p));
                
                end

                % error between average model adaptation and average measured adaptation
                model_error = mean(model-data,2);

                % log likelihood (probability of data given parameters)
                log_likelihood(run) = sum(-(log(2*pi*obj(1).sigma_motor_noise^2/P) + model_error.^2/(obj(1).sigma_motor_noise.^2/P))/2); % variance scaled by the number of participants

            end

            % negative of the log of the average likelihood across runs
            objective = -(log(1/obj(1).runs) + obj(1).log_sum_exp(log_likelihood));
        
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
            
            % is state feedback observed or not
            D.feedback_observed = ones(1,D.T);
            D.feedback_observed(isnan(obj.perturbations)) = 0;
            
            % self-transition bias
            D.kappa = obj.alpha_context*obj.rho_context/(1-obj.rho_context);
            
            % observation noise standard deviation
            D.sigma_observation_noise = sqrt(obj.sigma_sensory_noise^2 + obj.sigma_motor_noise^2);
            
            % matrix of context-dependent observation vectors
            D.H = eye(obj.max_contexts+1);

            % current trial
            D.t = 0;

            % number of contexts instantiated so far
            D.C = zeros(1,obj.particles);

            % context transition counts
            D.n_context = zeros(obj.max_contexts+1,obj.max_contexts+1,obj.particles);

            % sampled context
            D.context = ones(1,obj.particles); % treat trial 1 as a (context 1) self transition
            
            % do cues exist?
            if isempty(obj.cues) 
                D.cuesExist = 0;
            else
                D.cuesExist = 1;
                
                % number of contextual cues observed so far
                D.Q = 0;
                
                % cue emission counts
                D.n_cue = zeros(obj.max_contexts+1,max(obj.cues)+1,obj.particles);
            end

            % sufficient statistics for the parameters of the state dynamics
            % function
            D.dynamics_ss_1 = zeros(obj.max_contexts+1,obj.particles,2);
            D.dynamics_ss_2 = zeros(obj.max_contexts+1,obj.particles,2,2);

            % sufficient statistics for the parameters of the observation function
            D.bias_ss_1 = zeros(obj.max_contexts+1,obj.particles);
            D.bias_ss_2 = zeros(obj.max_contexts+1,obj.particles);

            % sample parameters from the prior
            D = sample_parameters(obj,D);
            
            % mean and variance of state (stationary distribution)
            D.state_filtered_mean = D.drift./(1-D.retention);
            D.state_filtered_var = obj.sigma_process_noise^2./(1-D.retention.^2);
            
        end
        
        function D = predict_context(obj,D)
            
            if ismember(D.t,obj.stationary_trials)
                % if some event (e.g. a working memory task) causes the context 
                % probabilities to be erased, set them to their stationary values
                for particle = 1:obj.particles
                    C = sum(D.local_transition_matrix(:,1,particle)>0);
                    T = D.local_transition_matrix(1:C,1:C,particle);        
                    D.prior_probabilities(1:C,particle) = obj.stationary_distribution(T);
                end
            else
                i = sub2ind(size(D.local_transition_matrix),repmat(D.context,[obj.max_contexts+1,1]),repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.particles,[obj.max_contexts+1,1]));
                D.prior_probabilities = D.local_transition_matrix(i);
            end

            if D.cuesExist
                i = sub2ind(size(D.local_cue_matrix),repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(obj.cues(D.t),[obj.max_contexts+1,obj.particles]),repmat(1:obj.particles,[obj.max_contexts+1,1]));
                D.probability_cue = D.local_cue_matrix(i);
                D.predicted_probabilities = D.prior_probabilities.*D.probability_cue;
                D.predicted_probabilities = D.predicted_probabilities./sum(D.predicted_probabilities,1);
            else
                D.predicted_probabilities = D.prior_probabilities;
            end
            
            if any(strcmp(obj.store,'Kalman_gain_given_cstar2'))
                if D.t > 1
                    [~,i] = max(D.predicted_probabilities,[],1);
                    i = sub2ind(size(D.Kalman_gains),i,1:obj.particles);
                    D.Kalman_gain_given_cstar2 = mean(D.Kalman_gains(i));
                end
            end
            
            if any(strcmp(obj.store,'state_given_cstar2'))
                if D.t > 1
                    [~,i] = max(D.predicted_probabilities,[],1);
                    i = sub2ind(size(D.state_mean),i,1:obj.particles);
                    D.state_given_cstar2 = mean(D.state_mean(i));
                end
            end
            
            if any(strcmp(obj.store,'predicted_probability_cstar3'))
                D.predicted_probability_cstar3 = mean(max(D.predicted_probabilities,[],1));
            end
            
        end

        function D = predict_states(obj,D)

            % propagate states
            D.state_mean = D.retention.*D.state_filtered_mean + D.drift;
            D.state_var = D.retention.^2.*D.state_filtered_var + obj.sigma_process_noise^2;

            % index of novel states
            i_new_x = sub2ind([obj.max_contexts+1,obj.particles],D.C+1,1:obj.particles);

            % novel states are distributed according to the stationary distribution
            D.state_mean(i_new_x) = D.drift(i_new_x)./(1-D.retention(i_new_x));
            D.state_var(i_new_x) = obj.sigma_process_noise^2./(1-D.retention(i_new_x).^2);

            % predict state (marginalise over contexts and particles)
            % mean of distribution
            D.average_state = sum(D.predicted_probabilities.*D.state_mean,'all')/obj.particles;

            if any(strcmp(obj.store,'explicit'))
                if D.t == 1
                    D.explicit = mean(D.state_mean(1,:));
                else
                    [~,i] = max(D.responsibilities,[],1);
                    i = sub2ind(size(D.state_mean),i,1:obj.particles);
                    D.explicit = mean(D.state_mean(i));
                end
            end
            
            if any(strcmp(obj.store,'state_given_cstar3'))
                [~,i] = max(D.predicted_probabilities,[],1);
                i = sub2ind(size(D.state_mean),i,1:obj.particles);
                D.state_given_cstar3 = mean(D.state_mean(i));
            end

        end
        
        function D = predict_state_feedback(obj,D)  

            % predict state feedback for each context
            D.state_feedback_mean = D.state_mean + D.bias;

            % variance of state feedback prediction for each context
            D.state_feedback_var = D.state_var + D.sigma_observation_noise^2;
            
            D = obj.compute_marginal_distribution(D);

            % predict state feedback (marginalise over contexts and particles)
            % mean of distribution
            D.motor_output = sum(D.predicted_probabilities.*D.state_feedback_mean,'all')/obj.particles;

            if any(strcmp(obj.store,'implicit'))
                D.implicit = D.motor_output - D.average_state;
            end

            % sensory and motor noise
            D.sensory_noise = obj.sigma_sensory_noise*randn;
            D.motor_noise = obj.sigma_motor_noise*randn;

            % state feedback
            D.state_feedback = obj.perturbations(D.t) + D.sensory_noise + D.motor_noise;

            % state feedback prediction error
            D.prediction_error = D.state_feedback - D.state_feedback_mean;

        end
        
        function D = resample_particles(obj,D)

            D.probability_state_feedback = normpdf(D.state_feedback,D.state_feedback_mean,sqrt(D.state_feedback_var)); % p(y_t|c_t)

            if D.feedback_observed(D.t)
                if D.cuesExist
                    p_c = log(D.prior_probabilities) + log(D.probability_cue) + log(D.probability_state_feedback); % log p(y_t,q_t,c_t)
                else
                    p_c = log(D.prior_probabilities) + log(D.probability_state_feedback); % log p(y_t,c_t)
                end
            else
                if D.cuesExist
                    p_c = log(D.prior_probabilities) + log(D.probability_cue);% log p(q_t,c_t)
                else
                    p_c = log(D.prior_probabilities); % log p(c_t)
                end
            end

            l_w = obj.log_sum_exp(p_c); % log p(y_t,q_t)

            p_c = p_c - l_w; % log p(c_t|y_t,q_t)

            % weights for resampling
            w = exp(l_w - obj.log_sum_exp(l_w'));

            % draw indices of particles to propagate
            if D.feedback_observed(D.t) || D.cuesExist
                D.i_resampled = obj.systematic_resampling(w);
            else
                D.i_resampled = 1:obj.particles;
            end
            
            % store variables of the predictive distributions (optional)
            % these variables are stored before resampling (so that they do not depend on the current state feedback)
            variables_stored_before_resampling = {'predicted_probabilities' 'state_feedback_mean' 'state_feedback_var' 'state_mean' 'state_var' 'Kalman_gain_given_cstar2' 'state_given_cstar2'};
            for i = 1:numel(obj.store)
                variable = obj.store{i};
                if any(strcmp(variable,variables_stored_before_resampling)) && isfield(D,variable)
                    D = obj.store_function(D,variable);
                end
            end

            % resample variables (particles)
            D.previous_context = D.context(D.i_resampled);
            D.prior_probabilities = D.prior_probabilities(:,D.i_resampled);
            D.predicted_probabilities = D.predicted_probabilities(:,D.i_resampled);
            D.responsibilities = exp(p_c(:,D.i_resampled)); % p(c_t|y_t,q_t)
            D.C = D.C(D.i_resampled);
            D.state_mean = D.state_mean(:,D.i_resampled);
            D.state_var = D.state_var(:,D.i_resampled);
            D.prediction_error = D.prediction_error(:,D.i_resampled);
            D.state_feedback_var = D.state_feedback_var(:,D.i_resampled);
            D.probability_state_feedback = D.probability_state_feedback(:,D.i_resampled);
            D.global_transition_probabilities = D.global_transition_probabilities(:,D.i_resampled);
            D.n_context = D.n_context(:,:,D.i_resampled);
            D.previous_state_filtered_mean = D.state_filtered_mean(:,D.i_resampled);
            D.previous_state_filtered_var = D.state_filtered_var(:,D.i_resampled);

            if D.cuesExist 
                D.global_cue_probabilities = D.global_cue_probabilities(:,D.i_resampled);
                D.n_cue = D.n_cue(:,:,D.i_resampled);
            end

            D.retention = D.retention(:,D.i_resampled);
            D.drift = D.drift(:,D.i_resampled);
            D.dynamics_ss_1 = D.dynamics_ss_1(:,D.i_resampled,:);
            D.dynamics_ss_2 = D.dynamics_ss_2(:,D.i_resampled,:,:);

            if obj.infer_bias
                D.bias = D.bias(:,D.i_resampled);
                D.bias_ss_1 = D.bias_ss_1(:,D.i_resampled);
                D.bias_ss_2 = D.bias_ss_2(:,D.i_resampled);
            end

        end

        function D = sample_context(obj,D)

            % sample the context
            D.context = sum(rand(1,obj.particles) > cumsum(D.responsibilities),1) + 1;

            % incremement the context count
            D.p_new_x = find(D.context > D.C);
            D.p_old_x = find(D.context <= D.C);
            D.C(D.p_new_x) = D.C(D.p_new_x) + 1;

            p_beta_x = D.p_new_x(D.C(D.p_new_x) ~= obj.max_contexts);
            i = sub2ind([obj.max_contexts+1,obj.particles],D.context(p_beta_x),p_beta_x);

            % sample the next stick-breaking weight
            beta = betarnd(1,obj.gamma_context*ones(1,numel(p_beta_x)));

            % update the global transition distribution
            D.global_transition_probabilities(i+1) = D.global_transition_probabilities(i).*(1-beta);
            D.global_transition_probabilities(i) = D.global_transition_probabilities(i).*beta;

            if D.cuesExist
                if obj.cues(D.t) > D.Q 
                    % increment the cue count
                    D.Q = D.Q + 1;

                    % sample the next stick-breaking weight
                    beta = betarnd(1,obj.gamma_cue*ones(1,obj.particles));

                    % update the global cue distribution
                    D.global_cue_probabilities(D.Q+1,:) = D.global_cue_probabilities(D.Q,:).*(1-beta);
                    D.global_cue_probabilities(D.Q,:) = D.global_cue_probabilities(D.Q,:).*beta;
                end

            end

        end

        function D = update_belief_about_states(~,D)

            D.Kalman_gains = D.state_var./D.state_feedback_var;
            if D.feedback_observed(D.t)
                D.state_filtered_mean = D.state_mean + D.Kalman_gains.*D.prediction_error.*D.H(D.context,:)';
                D.state_filtered_var = (1 - D.Kalman_gains.*D.H(D.context,:)').*D.state_var;
            else
                D.state_filtered_mean = D.state_mean;
                D.state_filtered_var = D.state_var;
            end

        end
        
        function D = sample_states(obj,D)

            n_new_x = numel(D.p_new_x);
            i_old_x = sub2ind([obj.max_contexts+1,obj.particles],D.context(D.p_old_x),D.p_old_x);
            i_new_x = sub2ind([obj.max_contexts+1,obj.particles],D.context(D.p_new_x),D.p_new_x);

            % for states that have been observed before, sample x_{t-1}, and then sample x_{t} given x_{t-1}

                % sample x_{t-1} using a fixed-lag (lag 1) forward-backward smoother
                g = D.retention.*D.previous_state_filtered_var./D.state_var;
                m = D.previous_state_filtered_mean + g.*(D.state_filtered_mean - D.state_mean);
                v = D.previous_state_filtered_var + g.*(D.state_filtered_var - D.state_var).*g;
                D.previous_x_dynamics = m + sqrt(v).*randn(obj.max_contexts+1,obj.particles);

                % sample x_t conditioned on x_{t-1} and y_t
                if D.feedback_observed(D.t)
                    w = (D.retention.*D.previous_x_dynamics + D.drift)./obj.sigma_process_noise^2 + D.H(D.context,:)'./D.sigma_observation_noise^2.*(D.state_feedback - D.bias);
                    v = 1./(1./obj.sigma_process_noise^2 + D.H(D.context,:)'./D.sigma_observation_noise^2);
                else
                    w = (D.retention.*D.previous_x_dynamics + D.drift)./obj.sigma_process_noise^2;
                    v = 1./(1./obj.sigma_process_noise^2);
                end
                D.x_dynamics = v.*w + sqrt(v).*randn(obj.max_contexts+1,obj.particles);

            % for novel states, sample x_t from the filtering distribution
            
                x_samp_novel = D.state_filtered_mean(i_new_x) + sqrt(D.state_filtered_var(i_new_x)).*randn(1,n_new_x);

            D.x_bias = [D.x_dynamics(i_old_x) x_samp_novel];
            D.i_observed = [i_old_x i_new_x];

        end
        
        function D = update_sufficient_statistics_for_parameters(obj,D)

            % update the sufficient statistics for the parameters of the 
            % global transition probabilities
            D = obj.update_sufficient_statistics_global_transition_probabilities(D);

            % update the sufficient statistics for the parameters of the 
            % global cue probabilities
            if D.cuesExist 
                D = obj.update_sufficient_statistics_global_cue_probabilities(D);
            end

            if D.t > 1
                % update the sufficient statistics for the parameters of the 
                % state dynamics function
                D = obj.update_sufficient_statistics_dynamics(D);
            end

            % update the sufficient statistics for the parameters of the 
            % observation function
            if obj.infer_bias && D.feedback_observed(D.t)
                D = obj.update_sufficient_statistics_bias(D);
            end

        end
         
        function D = sample_parameters(obj,D)

            % sample the global transition probabilities
            D = obj.sample_global_transition_probabilities(D);

            % update the local context transition probability matrix
            D = obj.update_local_transition_matrix(D);

            if D.cuesExist 
                % sample the global cue probabilities
                D = obj.sample_global_cue_probabilities(D);

                % update the local cue probability matrix
                D = obj.update_local_cue_matrix(D);
            end

            % sample the parameters of the state dynamics function
            D = obj.sample_dynamics(D);

            % sample the parameters of the observation function
            if obj.infer_bias
                D = obj.sample_bias(D);
            else
                D.bias = 0;
            end
            
        end
         
        function D = store_variables(obj,D)
            
            if any(strcmp(obj.store,'Kalman_gain_given_cstar1'))
                [~,i] = max(D.responsibilities,[],1);
                i = sub2ind(size(D.Kalman_gains),i,1:obj.particles);
                D.Kalman_gain_given_cstar1 = mean(D.Kalman_gains(i));
            end
            if any(strcmp(obj.store,'predicted_probability_cstar1'))
                [~,i] = max(D.responsibilities,[],1);
                i = sub2ind(size(D.predicted_probabilities),i,1:obj.particles);
                D.predicted_probability_cstar1 = mean(D.predicted_probabilities(i));
            end
            if any(strcmp(obj.store,'state_given_cstar1'))
                [~,i] = max(D.responsibilities,[],1);
                i = sub2ind(size(D.state_mean),i,1:obj.particles);
                D.state_given_cstar1 = mean(D.state_mean(i));
            end
             
            % store variables of the filtering distributions (optional)
            % these variables are stored after resampling (so that they depend on the current state feedback)
            variables_stored_before_resampling = {'predicted_probabilities' 'state_feedback_mean' 'state_feedback_var' 'state_mean' 'state_var' 'Kalman_gain_given_cstar2' 'state_given_cstar2'};
            for i = 1:numel(obj.store)
                variable = obj.store{i};
                if ~any(strcmp(variable,variables_stored_before_resampling))
                    D = obj.store_function(D,variable);
                end
            end
            
        end
         
        function D = update_sufficient_statistics_dynamics(obj,D)

            % augment the state vector: x_{t-1} --> [x_{t-1}; 1]
            x_a = ones(obj.max_contexts+1,obj.particles,2);
            x_a(:,:,1) = D.previous_x_dynamics;

            % identify states that are not novel
            I = reshape(sum(D.n_context,2),[obj.max_contexts+1,obj.particles]) > 0;

            SS = D.x_dynamics.*x_a; % x_t*[x_{t-1}; 1]
            D.dynamics_ss_1 = D.dynamics_ss_1 + SS.*I;

            SS = reshape(x_a,[obj.max_contexts+1,obj.particles,2]).*reshape(x_a,[obj.max_contexts+1,obj.particles,1,2]); % [x_{t-1}; 1]*[x_{t-1}; 1]'
            D.dynamics_ss_2 = D.dynamics_ss_2 + SS.*I;

        end
         
        function D = update_sufficient_statistics_bias(~,D)

            D.bias_ss_1(D.i_observed) = D.bias_ss_1(D.i_observed) + (D.state_feedback - D.x_bias); % y_t - x_t
            D.bias_ss_2(D.i_observed) = D.bias_ss_2(D.i_observed) + 1; % 1(c_t = j)
            
        end

        function D = update_sufficient_statistics_global_cue_probabilities(obj,D)

            i = sub2ind([obj.max_contexts+1,max(obj.cues)+1,obj.particles],D.context,obj.cues(D.t)*ones(1,obj.particles),1:obj.particles); % 1(c_t = j, q_t = k)
            D.n_cue(i) = D.n_cue(i) + 1;   

        end
         
        function D = update_sufficient_statistics_global_transition_probabilities(obj,D)

            i = sub2ind([obj.max_contexts+1,obj.max_contexts+1,obj.particles],D.previous_context,D.context,1:obj.particles); % 1(c_{t-1} = i, c_t = j)
            D.n_context(i) = D.n_context(i) + 1;
            
        end
         
        function D = sample_dynamics(obj,D)
            
            % prior mean and precision matrix
            dynamics_mean = [obj.prior_mean_retention 0]';
            dynamics_lambda = diag([obj.prior_precision_retention obj.prior_precision_drift]);

            % update the parameters of the posterior
            D.dynamics_covar = obj.per_slice_invert(dynamics_lambda + permute(reshape(D.dynamics_ss_2,[(obj.max_contexts+1)*obj.particles,2,2]),[2,3,1])/obj.sigma_process_noise^2);
            D.dynamics_mean = obj.per_slice_multiply(D.dynamics_covar,dynamics_lambda*dynamics_mean + reshape(D.dynamics_ss_1,[(obj.max_contexts+1)*obj.particles,2])'/obj.sigma_process_noise^2);
            
            % sample the parameters of the state dynamics function
            dynamics = obj.sample_from_truncated_bivariate_normal(D.dynamics_mean,D.dynamics_covar);
            D.retention = reshape(dynamics(1,:),[obj.max_contexts+1,obj.particles]);
            D.drift = reshape(dynamics(2,:),[obj.max_contexts+1,obj.particles]);
            
            % reshape
            D.dynamics_mean = reshape(D.dynamics_mean,[2,obj.max_contexts+1,obj.particles]);
            D.dynamics_covar = reshape(D.dynamics_covar,[2,2,obj.max_contexts+1,obj.particles]);

        end
         
        function D = sample_bias(obj,D)
            
            % prior mean
            bias_mean = 0;

            % update the parameters of the posterior
            D.bias_var = 1./(obj.prior_precision_bias + D.bias_ss_2./D.sigma_observation_noise^2);
            D.bias_mean = D.bias_var.*(obj.prior_precision_bias.*bias_mean + D.bias_ss_1./D.sigma_observation_noise^2);

            % sample the parameters of the observation function
            D.bias = obj.sample_from_univariate_normal(D.bias_mean,D.bias_var);

        end
         
        function D = sample_global_transition_probabilities(obj,D)

            if D.t == 0
                % global transition distribution
                D.global_transition_probabilities = zeros(obj.max_contexts+1,obj.particles);
                D.global_transition_probabilities(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.m_context = randnumtable(permute(obj.alpha_context*D.global_transition_probabilities,[3,1,2]) + D.kappa*eye(obj.max_contexts+1),D.n_context);

                % sample the number of tables in restaurant i considering dish j
                m_context_bar = D.m_context;
                if obj.rho_context > 0
                    i = sub2ind([obj.max_contexts+1,obj.max_contexts+1,obj.particles],repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.max_contexts+1,[obj.particles,1])',repmat(1:obj.particles,[obj.max_contexts+1,1]));
                    p = obj.rho_context./(obj.rho_context + D.global_transition_probabilities(D.m_context(i) ~= 0)*(1-obj.rho_context));
                    i = i(D.m_context(i) ~= 0);
                    m_context_bar(i) = D.m_context(i) - randbinom(p,D.m_context(i));
                end
                m_context_bar(1,1,(m_context_bar(1,1,:) == 0)) = 1;

                % sample beta
                i = find(D.C ~= obj.max_contexts);
                j = sub2ind([obj.max_contexts+1,obj.particles],D.C(i)+1,i);
                D.global_transition_posterior = squeeze(sum(m_context_bar,1));
                D.global_transition_posterior(j) = obj.gamma_context;
                D.global_transition_probabilities = obj.sample_from_dirichlet(D.global_transition_posterior);
            end

        end
         
        function D = sample_global_cue_probabilities(obj,D)

            if D.t == 0
                % global cue distribution
                D.global_cue_probabilities = zeros(max(obj.cues)+1,obj.particles);
                D.global_cue_probabilities(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.m_cue = randnumtable(repmat(permute(obj.alpha_cue.*D.global_cue_probabilities,[3,1,2]),[obj.max_contexts+1,1,1]),D.n_cue);

                % sample beta_e
                D.global_cue_posterior = reshape(sum(D.m_cue,1),[max(obj.cues)+1,obj.particles]);
                D.global_cue_posterior(D.Q+1,:) = obj.gamma_cue;
                D.global_cue_probabilities = obj.sample_from_dirichlet(D.global_cue_posterior);
            end

        end
         
        function D = update_local_cue_matrix(obj,D)

            D.local_cue_matrix = reshape(obj.alpha_cue.*D.global_cue_probabilities,[1,max(obj.cues)+1,obj.particles]) + D.n_cue;
            D.local_cue_matrix = D.local_cue_matrix./sum(D.local_cue_matrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.global_transition_probabilities,[obj.max_contexts+1,1,obj.particles]) > 0;
            D.local_cue_matrix = D.local_cue_matrix.*I;

        end
        
        function D = update_local_transition_matrix(obj,D)

            D.local_transition_matrix = reshape(obj.alpha_context*D.global_transition_probabilities,[1,obj.max_contexts+1,obj.particles]) + D.n_context + D.kappa*eye(obj.max_contexts+1);
            D.local_transition_matrix = D.local_transition_matrix./sum(D.local_transition_matrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.global_transition_probabilities,[obj.max_contexts+1,1,obj.particles]) > 0;
            D.local_transition_matrix = D.local_transition_matrix.*I;

            % compute stationary context probabilities if required
            if any(strcmp(obj.store,'stationary_probabilities')) && D.t > 0
                D.stationary_probabilities = zeros(obj.max_contexts+1,obj.particles);
                for particle = 1:obj.particles
                    C = D.C(particle);
                    T = D.local_transition_matrix(1:C+1,1:C+1,particle);
                    D.stationary_probabilities(1:C+1,particle) = obj.stationary_distribution(T);
                end
            end

        end
         
        function check_cue_labels(obj)

            % check cues are numbered according to the order they were presented in the experiment
            for trial = 1:numel(obj.perturbations)
                if trial == 1
                    if ~eq(obj.cues(trial),1)
                        obj.renumber_cues;
                        break
                    end
                else
                    if ~ismember(obj.cues(trial),obj.cues(1:trial-1)) && ~eq(obj.cues(trial),max(obj.cues(1:trial-1))+1)
                        obj.renumber_cues;
                        break
                    end
                end
            end
            
        end

        function renumber_cues(obj)

            cue_order = unique(obj.cues,'stable');
            if isrow(obj.cues)
                [obj.cues,~] = find(eq(obj.cues,cue_order'));
            else
                [obj.cues,~] = find(eq(obj.cues,cue_order')');
            end
            obj.cues = obj.cues';
            fprintf('Cues have been numbered according to the order they were presented in the experiment.\n')

        end
         
        function D = compute_marginal_distribution(obj,D)
             
            if any(strcmp(obj.store,'state_distribution'))
                % predict state (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.state_values,[1,1,numel(obj.state_values)]);
                mu = D.state_mean;
                sd = sqrt(D.state_var);
                D.state_distribution = sum(D.predicted_probabilities.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end
            if any(strcmp(obj.store,'bias_distribution'))
                % predict bias (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.bias_values,[1,1,numel(obj.bias_values)]);
                mu = D.bias_mean;
                sd = sqrt(D.bias_var);
                D.bias_distribution = sum(D.predicted_probabilities.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end
            if any(strcmp(obj.store,'state_feedback_distribution'))
                % predict state feedback (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.state_feedback_values,[1,1,numel(obj.state_feedback_values)]);
                mu = D.state_feedback_mean;
                sd = sqrt(D.state_feedback_var);
                D.state_feedback_distribution = sum(D.predicted_probabilities.*normpdf(x,mu,sd),[1,2])/obj.particles;
            end

        end
         
        function l = log_sum_exp(~,logP)

            m = max(logP,[],1);
            l = m + log(sum(exp(logP - m),1));
            
        end
         
        function L = per_slice_cholesky(~,V)

            % perform cholesky decomposition on each 2 x 2 slice of array V to
            % obtain lower triangular matrix L

            L = zeros(size(V));
            L(1,1,:) = sqrt(V(1,1,:));
            L(2,1,:) = V(2,1,:)./L(1,1,:);
            L(2,2,:) = sqrt(V(2,2,:) - L(2,1,:).^2);
            
        end

        function L_inverse = per_slice_invert(~,L)

            % invert each 2 x 2 slice of array L

            L_determinant = L(1,1,:).*L(2,2,:)-L(1,2,:).*L(2,1,:);
            L_inverse = [L(2,2,:) L(1,2,:); L(2,1,:) L(1,1,:)].*[1 -1; -1 1]./L_determinant;
            
        end

        function C = per_slice_multiply(~,A,B)

            % per slice matrix multiplication
            C = squeeze(sum(A.*permute(B,[3,1,2]),2));

        end
        
        function x = sample_from_dirichlet(~,A)

            % sample categorical parameter from dirichlet distribution
            x = randgamma(A);
            x = x./sum(x,1);

        end
        
        function x = sample_from_truncated_bivariate_normal(obj,mu,V)

            % perform cholesky decomposition on the covariance matrix V
            V_cholesky = obj.per_slice_cholesky(V);

            % truncation bounds for the state retention factor
            aMin = 0;
            aMax = 1;

            % equivalent truncation bounds for the standard normal distribution
            l = [(aMin-mu(1,:))./squeeze(V_cholesky(1,1,:))'; -Inf*ones(1,size(V,3))];
            u = [(aMax-mu(1,:))./squeeze(V_cholesky(1,1,:))'; Inf*ones(1,size(V,3))];

            % transform samples from the truncated standard normal distribution to
            % samples from the desired truncated normal distribution
            x = mu + obj.per_slice_multiply(V_cholesky,reshape(trandn(l,u),[2,size(V,3)]));

        end

        function x = sample_from_univariate_normal(obj,mu,sigma)

            x = mu + sqrt(sigma).*randn(obj.max_contexts+1,obj.particles);

        end
        
        function p = stationary_distribution(~,T)
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

        function p = systematic_resampling(~,w)

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
        
        function D = store_function(~,D,variable)

            if any(strcmp(variable,{'Kalman_gain_given_cstar2' 'state_given_cstar2'}))
                store_on = 'previous_trial';
            else
                store_on = 'current_trial';
            end

            % preallocate memory for variables to be stored
            if (D.t == 1 && strcmp(store_on,'current_trial')) || (D.t == 2 && strcmp(store_on,'previous_trial'))
                s = [eval(sprintf('size(D.%s)',variable)) D.T];
                eval(sprintf('D.stored.%s = squeeze(NaN(s));',variable))
            end

            % store variables
            dims = eval(sprintf('sum(size(D.%s)>1)',variable));
            if strcmp(store_on,'current_trial')
                trial = 'D.t';
            elseif strcmp(store_on,'previous_trial')
                trial = 'D.t-1';
            end
            eval(sprintf('D.stored.%s(%s%s) = D.%s;',variable,repmat(':,',[1,dims]),trial,variable))

        end
        
        function P = plot_COIN(obj,S)
            
            variables_that_require_context_relabelling = {'state_given_context' 'predicted_probabilities' 'responsibilities' 'stationary_probabilities' 'retention_given_context'...
            'drift_given_context' 'bias_given_context' 'global_transition_probabilities' 'local_transition_probabilities' 'global_cue_probabilities' 'local_cue_probabilities'};
            for i = 1:numel(variables_that_require_context_relabelling)
                if eval(sprintf('obj.plot_%s == true',variables_that_require_context_relabelling{i}))
                    [P,S,optAssignment,from_unique,cSeq,C] = obj.find_optimal_context_labels(S);
                    [P,~] = obj.compute_variables_for_plotting(P,S,optAssignment,from_unique,cSeq,C);
                    break
                elseif i == numel(variables_that_require_context_relabelling)
                    P = obj.preallocate_memory_for_plot_variables([]);
                    P = obj.integrate_over_runs(P,S);
                end
            end
            obj.generate_figures(P);
            
        end
        
        function [P,S,optimal_assignment,from_unique,context_sequence,C] = find_optimal_context_labels(obj,S)
            
            i_resampled = obj.resample_indices(S);
            
            context_sequence = obj.context_sequences(S,i_resampled);

            [C,~,~,P.mode_number_of_contexts] = obj.posterior_number_of_contexts(context_sequence,S);
            
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
            optimal_assignment = cell(1,T);
            for trial = 1:T

                if ~mod(trial,50)
                    fprintf('Finding optimal context labels (trial = %d).\n',trial)
                end

                    % exclude sequences for which C > max(P.mode_number_of_contexts) as 
                    % these sequences (and their descendents) will never be 
                    % analysed
                    f{trial} = find(C(:,trial) <= max(P.mode_number_of_contexts));

                    % identify unique sequences (to avoid performing the
                    % same computations multiple times)
                    [unique_sequences,to_unique{trial},from_unique{trial}] = unique(context_sequence{trial}(f{trial},:),'rows');

                    % number of unique sequences
                    n_sequences = size(unique_sequences,1);

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
                        [i,~] = find(f{trial-1} == i_resampled(f{trial}(to_unique{trial}),trial)');
                        parent = from_unique{trial-1}(i);

                        % pass Hamming distances from parents to children
                        i = sub2ind(size(H),repmat(parent,[1,n_sequences,nPerm]),repmat(parent',[n_sequences,1,nPerm]),repmat(permute(1:nPerm,[1,3,2]),[n_sequences,n_sequences,1]));
                        H = H(i);
                        
                        % recursively update Hamming distances
                        % dimension 3 of H considers all possible label permutations
                        for seq = 1:n_sequences
                            H(seq:end,seq,:) = H(seq:end,seq,:) + double(unique_sequences(seq,end) ~= L(unique_sequences(seq:end,end),:,:));
                            H(seq,seq:end,:) = H(seq:end,seq,:); % Hamming distance is symmetric
                        end
                    end

                    % compute the Hamming distance between each pair of
                    % sequences (after optimally permuting labels)
                    Hopt = min(H,[],3);
                    
                    % count the number of times each unique sequence occurs
                    sequence_count = sum(from_unique{trial}(idx) == 1:size(unique_sequences,1),1);

                    % compute the mean optimal Hamming distance of each 
                    % sequence to all other sequences. the distance from
                    % sequence i to sequence j is weighted by the number of
                    % times sequence j occurs. if i == j, this weight is 
                    % reduced by 1 so that the distance from one instance 
                    % of sequence i to itself is ignored
                    H_mean = mean(Hopt.*(sequence_count-eye(n_sequences)),2);
                    
                    % assign infinite distance to invalid sequences (i.e.
                    % sequences for which the number of contexts is not equal 
                    % to the most common number of contexts)
                    H_mean(sequence_count == 0) = Inf;

                    % find the index of the typical sequence (the sequence
                    % with the minimum mean optimal Hamming distance to all
                    % other sequences)
                    [~,i] = min(H_mean);

                    % typical context sequence
                    % n.b. the typical context sequence does not consider the potentially nonuniform nature of run weights
                    typical_sequence = unique_sequences(i,:);
                    
                    % store the optimal permutation of labels for each sequence 
                    % with respect to the typical sequence
                    [~,j] = min(H(i,:,:),[],3);
                    optimal_assignment{trial} = permute(L(1:P.mode_number_of_contexts(trial),:,j),[3,1,2]);

            end

        end
        
        function i_resampled = resample_indices(obj,S)
            
            % number of trials
            T = numel(obj.perturbations);
            
            i_resampled = zeros(obj.particles*obj.runs,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                i_resampled(p,:) = obj.particles*(run-1) + S.runs{run}.i_resampled;
            end
            
        end
        
        function context_sequence = context_sequences(obj,S,i_resampled)
            
            % number of trials
            T = numel(obj.perturbations);
            
            context_sequence = cell(1,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                for trial = 1:T
                    if run == 1
                        context_sequence{trial} = zeros(obj.particles*obj.runs,trial);
                    end
                    if trial > 1
                        context_sequence{trial}(p,1:trial-1) = context_sequence{trial-1}(p,:);
                        context_sequence{trial}(p,:) = context_sequence{trial}(i_resampled(p,trial),:);
                    end
                    context_sequence{trial}(p,trial) = S.runs{run}.context(:,trial);
                end
            end
            
        end
        
        function [C,posterior,posterior_mean,posterior_mode] = posterior_number_of_contexts(obj,context_sequence,S)
            
            % number of trials
            T = numel(obj.perturbations);
            
            % number of contexts
            C = zeros(obj.particles*obj.runs,T);
            for run = 1:obj.runs
                p = obj.particles*(run-1) + (1:obj.particles);
                for trial = 1:T
                    C(p,trial) = max(context_sequence{trial}(p,:),[],2);
                end
            end

            particle_weight = repelem(S.weights,obj.particles)/obj.particles;
            if isrow(particle_weight)
                particle_weight = particle_weight';
            end

            posterior = zeros(obj.max_contexts+1,T);
            posterior_mean = zeros(1,T);
            posterior_mode = zeros(1,T);
            for time = 1:T

                for context = 1:max(C(:,time),[],'all')
                    posterior(context,time) = sum((C(:,time) == context).*particle_weight);
                end
                posterior_mean(time) = (1:obj.max_contexts+1)*posterior(:,time);
                [~,posterior_mode(time)] = max(posterior(:,time));

            end
            
        end
        
        function [P,S] = compute_variables_for_plotting(obj,P,S,optimal_assignment,from_unique,context_sequence,C)

            % number of trials
            T = numel(obj.perturbations);
            
            P = obj.preallocate_memory_for_plot_variables(P);
            
            n_particles_used = zeros(T,obj.runs);
            for trial = 1:T
                if ~mod(trial,50)
                    fprintf('Permuting context labels (trial = %d).\n',trial)
                end
                
                % cumulative number of particles for which C <= max(P.mode_number_of_contexts)
                N = 0;

                for run = 1:obj.runs
                    % indices of particles of the current run
                    p = obj.particles*(run-1) + (1:obj.particles);
                    
                    % indices of particles that are either valid now or 
                    % could be valid in the future: C <= max(P.mode_number_of_contexts)
                    valid_future = find(C(p,trial) <= max(P.mode_number_of_contexts));
                    
                    % indices of particles that are valid now: C == P.mode_number_of_contexts(trial)
                    valid_now = find(C(p,trial) == P.mode_number_of_contexts(trial))';
                    n_particles_used(trial,run) = numel(valid_now);
                    
                    if ~isempty(valid_now)
                        for particle = valid_now
                            % index of the optimal label permutations of the
                            % current particle
                            i = N + find(particle == valid_future);
                            
                            % is the latest context a novel context?
                            % this is needed to store novel context probabilities
                            context_trajectory = context_sequence{trial}(p(particle),:);
                            novel_context = context_trajectory(trial) > max(context_trajectory(1:trial-1));
                            
                            S = obj.relabel_context_variables(S,optimal_assignment{trial}(from_unique{trial}(i),:),novel_context,particle,trial,run);
                        end
                        P = obj.integrate_over_particles(S,P,valid_now,trial,run);
                    end
                    N = N + numel(valid_future);
                end
            end

            P = obj.integrate_over_runs(P,S);

            P = obj.normalise_relabelled_variables(P,n_particles_used,S);

            if obj.plot_state_given_context
                % the predicted state distribution for a novel context is the marginal
                % stationary distribution of the state after integrating out 
                % the drift and retention parameters under the prior
                P.state_given_novel_context = repmat(nanmean(P.state_given_context(:,:,end),2),[1,T]);
                P.state_given_context = P.state_given_context(:,:,1:end-1);
            end

        end

        function S = relabel_context_variables(obj,S,optimal_assignment,novel_context,particle,trial,run)

            C = numel(optimal_assignment);
            
            % number of trials
            T = numel(obj.perturbations);

            % predictive distributions
            if trial < T
                if obj.plot_state_given_context
                    S.runs{run}.state_mean(optimal_assignment,particle,trial+1) = S.runs{run}.state_mean(1:C,particle,trial+1);
                    S.runs{run}.state_var(optimal_assignment,particle,trial+1) = S.runs{run}.state_var(1:C,particle,trial+1);
                end
                if obj.plot_predicted_probabilities
                    S.runs{run}.predicted_probabilities(optimal_assignment,particle,trial+1) = S.runs{run}.predicted_probabilities(1:C,particle,trial+1);
                end
            end

            if obj.plot_responsibilities
                if trial == 1 || novel_context 
                    S.runs{run}.responsibilities([1:C-1 C+1],particle,trial) = S.runs{run}.responsibilities(optimal_assignment,particle,trial);
                    S.runs{run}.responsibilities(C,particle,trial) = NaN;
                else
                    S.runs{run}.responsibilities(1:C+1,particle,trial) = S.runs{run}.responsibilities([optimal_assignment C+1],particle,trial);
                end
            end
            if obj.plot_stationary_probabilities
                S.runs{run}.stationary_probabilities(optimal_assignment,particle,trial) = S.runs{run}.stationary_probabilities(1:C,particle,trial);
            end
            if obj.plot_retention_given_context || obj.plot_drift_given_context
                S.runs{run}.dynamics_mean(:,optimal_assignment,particle,trial) = S.runs{run}.dynamics_mean(:,1:C,particle,trial);
                S.runs{run}.dynamics_covar(:,:,optimal_assignment,particle,trial) = S.runs{run}.dynamics_covar(:,:,1:C,particle,trial);
            end
            if obj.plot_bias_given_context
                S.runs{run}.bias_mean(optimal_assignment,particle,trial) = S.runs{run}.bias_mean(1:C,particle,trial);
                S.runs{run}.bias_var(optimal_assignment,particle,trial) = S.runs{run}.bias_var(1:C,particle,trial);
            end
            if obj.plot_global_transition_probabilities
                S.runs{run}.global_transition_posterior(optimal_assignment,particle,trial) = S.runs{run}.global_transition_posterior(1:C,particle,trial);
            end
            if obj.plot_local_transition_probabilities
                S.runs{run}.local_transition_matrix(1:C,1:C+1,particle,trial) = ...
                obj.permute_transition_matrix_columns_and_rows(S.runs{run}.local_transition_matrix(1:C,1:C+1,particle,trial),optimal_assignment);
            end
            if obj.plot_local_cue_probabilities
                S.runs{run}.local_cue_matrix(optimal_assignment,:,particle,trial) = S.runs{run}.local_cue_matrix(1:C,:,particle,trial);
            end

        end
        
        function P = permute_transition_matrix_columns_and_rows(~,T,optimal_assignment)

                C = numel(optimal_assignment);

                [i_map,~] = find(optimal_assignment' == 1:C); % inverse mapping

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
            if obj.plot_Kalman_gain_given_cstar1
                P.Kalman_gain_given_cstar1 = NaN(T,obj.runs);
            end
            if obj.plot_predicted_probability_cstar1
                P.predicted_probability_cstar1 = NaN(T,obj.runs);
            end
            if obj.plot_state_given_cstar1
                P.state_given_cstar1 = NaN(T,obj.runs);
            end
            if obj.plot_Kalman_gain_given_cstar2
                P.Kalman_gain_given_cstar2 = NaN(T,obj.runs);
            end
            if obj.plot_state_given_cstar2
                P.state_given_cstar2 = NaN(T,obj.runs);
            end 
            if obj.plot_predicted_probability_cstar3
                P.predicted_probability_cstar3 = NaN(T,obj.runs);
            end
            if obj.plot_state_given_cstar3
                P.state_given_cstar3 = NaN(T,obj.runs);
            end 
            P.average_state_feedback = NaN(T,obj.runs);  

        end
        
        function P = integrate_over_particles(obj,S,P,particles,trial,run)

            C = P.mode_number_of_contexts(trial);
            novel_context = max(P.mode_number_of_contexts)+1;
            
            % number of trials
            T = numel(obj.perturbations);

            % predictive distributions
            if trial < T
                if obj.plot_state_given_context
                    mu = permute(S.runs{run}.state_mean(1:C+1,particles,trial+1),[3,2,1]);
                    sd = permute(sqrt(S.runs{run}.state_var(1:C+1,particles,trial+1)),[3,2,1]);
                    P.state_given_context(:,trial+1,[1:C novel_context],run) = sum(normpdf(obj.state_values',mu,sd),2);
                end
                if obj.plot_predicted_probabilities
                    P.predicted_probabilities(trial+1,[1:C novel_context],run) = sum(S.runs{run}.predicted_probabilities(1:C+1,particles,trial+1),2);
                end
            end

            if obj.plot_responsibilities
                P.responsibilities(trial,[1:C novel_context],run) = obj.sum_along_dimension(S.runs{run}.responsibilities(1:C+1,particles,trial),2);
            end
            if obj.plot_stationary_probabilities
                P.stationary_probabilities(trial,[1:C novel_context],run) = sum(S.runs{run}.stationary_probabilities(1:C+1,particles,trial),2);
            end
            if obj.plot_retention_given_context 
                mu = permute(S.runs{run}.dynamics_mean(1,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S.runs{run}.dynamics_covar(1,1,1:C,particles,trial)),[1,4,3,2]);
                P.retention_given_context(:,trial,1:C,run) = sum(normpdf(obj.retention_values',mu,sd),2);
            end
            if obj.plot_drift_given_context
                mu = permute(S.runs{run}.dynamics_mean(2,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S.runs{run}.dynamics_covar(2,2,1:C,particles,trial)),[1,4,3,2]);
                P.drift_given_context(:,trial,1:C,run) = sum(normpdf(obj.drift_values',mu,sd),2);  
            end
            if obj.plot_bias_given_context
                mu = permute(S.runs{run}.bias_mean(1:C,particles,trial),[3,2,1]);
                sd = permute(sqrt(S.runs{run}.bias_var(1:C,particles,trial)),[3,2,1]);
                P.bias_given_context(:,trial,1:C,run) = sum(normpdf(obj.bias_values',mu,sd),2);  
            end
            if obj.plot_global_transition_probabilities
                alpha = S.runs{run}.global_transition_posterior(1:C+1,particles,trial);
                P.global_transition_probabilities(trial,[1:C novel_context],run) = sum(alpha./sum(alpha,1),2);
            end
            if obj.plot_local_transition_probabilities
                P.local_transition_probabilities(1:C,[1:C novel_context],trial,run) = sum(S.runs{run}.local_transition_matrix(1:C,1:C+1,particles,trial),3);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities(1:C,[1:max(obj.cues(1:trial)) max(obj.cues)+1],trial,run) = sum(S.runs{run}.local_cue_matrix(1:C,1:max(obj.cues(1:trial))+1,particles,trial),3);
            end

        end
        
        function X = sum_along_dimension(~,X,dim)

            % find elements that are NaN throughout dimension dim
            i = all(isnan(X),dim);

            % sum over dimension dim of X
            X = sum(X,dim,'omitnan');

            % elements that are NaN throughout dimension dim should remain NaN, not 0
            X(i) = NaN;

        end
        
        function P = integrate_over_runs(obj,P,S)

            if obj.plot_state_given_context
                P.state_given_context = obj.weighted_sum_along_dimension(P.state_given_context,S,4);
            end
            if obj.plot_predicted_probabilities
                P.predicted_probabilities = obj.weighted_sum_along_dimension(P.predicted_probabilities,S,3);
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
                for run = 1:obj.runs
                    for trial = 1:numel(obj.perturbations)
                        alpha = S.runs{run}.global_cue_posterior(1:max(obj.cues(1:trial))+1,:,trial);
                        P.global_cue_probabilities(trial,[1:max(obj.cues(1:trial)) max(obj.cues)+1],run) = sum(alpha./sum(alpha,1),2);
                    end
                end
                P.global_cue_probabilities = obj.weighted_sum_along_dimension(P.global_cue_probabilities,S,3);
            end
            if obj.plot_local_cue_probabilities
                P.local_cue_probabilities = obj.weighted_sum_along_dimension(P.local_cue_probabilities,S,4);
            end
            
            if obj.plot_state
                for run = 1:obj.runs
                    P.state(:,:,run) = S.runs{run}.state_distribution;
                end
                P.state = obj.weighted_sum_along_dimension(P.state,S,3);
            end
            if obj.plot_average_state || obj.plot_state
                for run = 1:obj.runs
                    P.average_state(:,run) = S.runs{run}.average_state;
                end
                P.average_state = obj.weighted_sum_along_dimension(P.average_state,S,2);
            end
            if obj.plot_bias
                for run = 1:obj.runs
                    P.bias(:,:,run) = S.runs{run}.bias_distribution;
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
                    P.state_feedback(:,:,run) = S.runs{run}.state_feedback_distribution;
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
            if obj.plot_Kalman_gain_given_cstar1
                for run = 1:obj.runs
                    P.Kalman_gain_given_cstar1(:,run) = S.runs{run}.Kalman_gain_given_cstar1;
                end
                P.Kalman_gain_given_cstar1 = obj.weighted_sum_along_dimension(P.Kalman_gain_given_cstar1,S,2);
            end
            if obj.plot_predicted_probability_cstar1
                for run = 1:obj.runs
                    P.predicted_probability_cstar1(:,run) = S.runs{run}.predicted_probability_cstar1;
                end
                P.predicted_probability_cstar1 = obj.weighted_sum_along_dimension(P.predicted_probability_cstar1,S,2);
            end
            if obj.plot_state_given_cstar1
                for run = 1:obj.runs
                    P.state_given_cstar1(:,run) = S.runs{run}.state_given_cstar1;
                end
                P.state_given_cstar1 = obj.weighted_sum_along_dimension(P.state_given_cstar1,S,2);
            end
            if obj.plot_Kalman_gain_given_cstar2
                for run = 1:obj.runs
                    P.Kalman_gain_given_cstar2(:,run) = S.runs{run}.Kalman_gain_given_cstar2;
                end
                P.Kalman_gain_given_cstar2 = obj.weighted_sum_along_dimension(P.Kalman_gain_given_cstar2,S,2);
            end
            if obj.plot_state_given_cstar2
                for run = 1:obj.runs
                    P.state_given_cstar2(:,run) = S.runs{run}.state_given_cstar2;
                end
                P.state_given_cstar2 = obj.weighted_sum_along_dimension(P.state_given_cstar2,S,2);
            end 
            if obj.plot_predicted_probability_cstar3
                for run = 1:obj.runs
                    P.predicted_probability_cstar3(:,run) = S.runs{run}.predicted_probability_cstar3;
                end
                P.predicted_probability_cstar3 = obj.weighted_sum_along_dimension(P.predicted_probability_cstar3,S,2);
            end
            if obj.plot_state_given_cstar3
                for run = 1:obj.runs
                    P.state_given_cstar3(:,run) = S.runs{run}.state_given_cstar3;
                end
                P.state_given_cstar3 = obj.weighted_sum_along_dimension(P.state_given_cstar3,S,2);
            end 
            for run = 1:obj.runs
                P.average_state_feedback(:,run) = S.runs{run}.motor_output;
            end
            P.average_state_feedback = obj.weighted_sum_along_dimension(P.average_state_feedback,S,2);

        end
        
        function X = weighted_sum_along_dimension(~,X,S,dim)

            % find elements that are NaN throughout dimension dim
            i = all(isnan(X),dim);

            % sum over dimension dim of X with weights w
            X = sum(X.*reshape(S.weights,[ones(1,dim-1) numel(S.weights)]),dim,'omitnan');

            % elements that are NaN throughout dimension dim should remain NaN, not 0
            X(i) = NaN;

        end
        
        function P = normalise_relabelled_variables(obj,P,n_particles_used,S)
            
            % number of trials
            T = numel(obj.perturbations);

            % normalisation constant
            Z = sum(n_particles_used.*S.weights,2);

            if obj.plot_state_given_context
                P.state_given_context(:,2:end,:) = P.state_given_context(:,2:end,:)./Z(1:end-1)';
            end
            if obj.plot_predicted_probabilities
                P.predicted_probabilities(2:end,:) = P.predicted_probabilities(2:end,:)./Z(1:end-1);
            end
            if obj.plot_responsibilities
                P.responsibilities = P.responsibilities./Z;
                P.novel_context_responsibility = P.responsibilities(:,end);
                P.known_context_responsibilities = P.responsibilities(:,1:end-1);
                P = rmfield(P,'responsibilities');
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
            if obj.plot_global_cue_probabilities
                P.global_cue_probabilities = P.global_cue_probabilities./obj.particles;
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
        
        function YTicksPixels = map_to_pixel_space(~,n_pixels,lims,y_ticks)

            % map points to pixel space

            % imagesc plots pixels of size 1 centered on the integers (e.g. the first 
            % pixel is centered on 1 and spans from 0.5 to 1.5)

            lims = sort(lims,'descend');
            YTicksPixels = 1 + (n_pixels-1)*((y_ticks-lims(1))/(lims(2)-lims(1)));

        end
        
        function generate_figures(obj,P)

            C = obj.colours;
            
            line_width = 2;
            font_size = 15;
            
            % number of trials
            T = numel(obj.perturbations);
            
            if obj.plot_state_given_context
                figure
                y_lims = obj.state_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.state_given_context,y_lims,y_ticks,C.contexts(1:max(P.mode_number_of_contexts),:))
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('state | context')
                xlabel('trial')
                figure
                y_lims = obj.state_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.state_given_novel_context,y_lims,y_ticks,C.new_context)
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('state | novel context')
                xlabel('trial')
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
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('predicted probability')
                xlabel('trial')
            end
            if obj.plot_responsibilities
                figure
                hold on
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.known_context_responsibilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('known context responsibility')
                xlabel('trial')
                figure
                plot(P.novel_context_responsibility,'Color',C.new_context,'LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('novel context responsibility')
                xlabel('trial')
            end
            if obj.plot_stationary_probabilities
                figure
                hold on
                plot(P.stationary_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.stationary_probabilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('stationary context probability')
                xlabel('trial')
            end
            if obj.plot_retention_given_context 
                figure
                y_lims = obj.retention_values([1 end]);
                y_ticks = [0 obj.retention_values([1 end])];
                obj.plot_image(P.retention_given_context,y_lims,y_ticks,C.contexts)
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('retention | context')
                xlabel('trial')
            end
            if obj.plot_drift_given_context
                figure
                y_lims = obj.drift_values([1 end]);
                y_ticks = [0 obj.drift_values([1 end])];
                obj.plot_image(P.drift_given_context,y_lims,y_ticks,C.contexts)
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('drift | context')
                xlabel('trial')
            end
            if obj.plot_bias_given_context
                figure
                y_lims = obj.bias_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.bias_given_context,y_lims,y_ticks,C.contexts)
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('bias | context')
                xlabel('trial')
            end
            if obj.plot_global_transition_probabilities
                figure
                hold on
                plot(P.global_transition_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for context = 1:max(P.mode_number_of_contexts)
                    plot(P.global_transition_probabilities(:,context),'Color',C.contexts(context,:),'LineWidth',line_width)
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('global transition probability')
                xlabel('trial')
            end
            if obj.plot_local_transition_probabilities
                for from_context = 1:max(P.mode_number_of_contexts)
                    figure
                    hold on
                    tmp = {'to novel context'};
                    plot(squeeze(P.local_transition_probabilities(from_context,end,:)),'Color',C.new_context,'LineWidth',line_width)
                    for to_context = 1:max(P.mode_number_of_contexts)
                        plot(squeeze(P.local_transition_probabilities(from_context,to_context,:)),'Color',C.contexts(to_context,:),'LineWidth',line_width)
                        tmp = cat(2,tmp,sprintf('to context %d',to_context));
                    end
                    title(sprintf('\\color[rgb]{%s}from context %d',num2str(C.contexts(from_context,:)),from_context))
                    axis([0 T -0.1 1.1])
                    set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                    ylabel('local transition probability')
                    xlabel('trial')
                    legend(tmp,'location','best','box','off')
                end
            end
            if obj.plot_global_cue_probabilities
                figure
                hold on
                tmp = {'novel cue'};
                plot(P.global_cue_probabilities(:,end),'Color',C.new_context,'LineWidth',line_width)
                for cue = 1:max(obj.cues)
                    plot(P.global_cue_probabilities(:,cue),'Color',C.cues(cue,:),'LineWidth',line_width)
                    tmp = cat(2,tmp,sprintf('cue %d',cue));
                end
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('global cue probability')
                xlabel('trial')
                legend(tmp,'location','best','box','off')
            end
            if obj.plot_local_cue_probabilities
                for context = 1:max(P.mode_number_of_contexts)
                    figure
                    hold on
                    tmp = {'novel cue'};
                    plot(squeeze(P.local_cue_probabilities(context,end,:)),'Color',C.new_context,'LineWidth',line_width)
                    for cue = 1:max(obj.cues)
                        plot(squeeze(P.local_cue_probabilities(context,cue,:)),'Color',C.cues(cue,:),'LineWidth',line_width)
                        tmp = cat(2,tmp,sprintf('cue %d',cue));
                    end
                    title(sprintf('\\color[rgb]{%s}context %d',num2str(C.contexts(context,:)),context))
                    axis([0 T -0.1 1.1])
                    set(gca,'YTick',[0 1],'XTick',[0 T],'FontSize',font_size)
                    ylabel('local cue probability')
                    xlabel('trial')
                    legend(tmp,'location','best','box','off')
                end
            end
            if obj.plot_state
                figure
                y_lims = obj.state_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.state,y_lims,y_ticks,C.marginal)
                hold on
                n_pixels = numel(obj.state_values);
                plot(obj.map_to_pixel_space(n_pixels,y_lims,P.average_state),'Color',C.mean_of_marginal,'LineWidth',line_width)
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('state')
                xlabel('trial')
            end
            if obj.plot_average_state
                figure
                plot(P.average_state,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('average state')
                xlabel('trial')
            end
            if obj.plot_bias
                figure
                y_lims = obj.bias_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.bias,y_lims,y_ticks,C.marginal)
                hold on
                n_pixels = numel(obj.bias_values);
                plot(obj.map_to_pixel_space(n_pixels,y_lims,P.average_bias),'Color',C.mean_of_marginal,'LineWidth',line_width)  
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('bias')
                xlabel('trial')
            end
            if obj.plot_average_bias
                figure
                plot(P.average_bias,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('average bias')
                xlabel('trial')
            end
            if obj.plot_state_feedback
                figure
                y_lims = obj.state_feedback_values([1 end]);
                y_ticks = [-1 0 1];
                obj.plot_image(P.state_feedback,y_lims,y_ticks,C.marginal)
                hold on
                n_pixels = numel(obj.state_feedback_values);
                plot(obj.map_to_pixel_space(n_pixels,y_lims,P.average_state_feedback ),'Color',C.mean_of_marginal,'LineWidth',line_width)  
                set(gca,'FontSize',font_size,'XTick',[1 T],'XTickLabels',[1 T])
                xlim([1 T])
                ylabel('state feedback')
                xlabel('trial')
            end
            if obj.plot_explicit_component
                figure
                plot(P.explicit_component,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('explicit component of adaptation')
                xlabel('trial')
            end
            if obj.plot_implicit_component
                figure
                plot(P.implicit_component,'Color',C.mean_of_marginal,'LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('implicit component of adaptation')
                xlabel('trial')
            end
            if obj.plot_Kalman_gain_given_cstar1
                figure
                plot(P.Kalman_gain_given_cstar1,'Color','k','LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('Kalman gain | c^*')
                title('c^* is the context with the highest responsibility')
                xlabel('trial')
            end
            if obj.plot_predicted_probability_cstar1
                figure
                plot(P.predicted_probability_cstar1,'Color','k','LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('predicted probability (c^*)')
                title('c^* is the context with the highest responsibility')
                xlabel('trial')
            end
            if obj.plot_state_given_cstar1
                figure
                plot(P.state_given_cstar1,'Color','k','LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('E[ state | c^*]')
                title('c^* is the context with the highest responsibility')
                xlabel('trial')
            end
            if obj.plot_Kalman_gain_given_cstar2
                figure
                plot(P.Kalman_gain_given_cstar2,'Color','k','LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('Kalman gain | c^*')
                title('c^* is the context with the highest predicted probability on the next trial')
                xlabel('trial')
            end
            if obj.plot_state_given_cstar2
                figure
                plot(P.state_given_cstar2,'Color','k','LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('E[ state | c^*]')
                title('c^* is the context with the highest predicted probability on the next trial')
                xlabel('trial')
            end 
            if obj.plot_predicted_probability_cstar3
                figure
                plot(P.predicted_probability_cstar3,'Color','k','LineWidth',line_width)
                axis([0 T -0.1 1.1])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('predicted probability (c^*)')
                title('c^* is the context with the highest predicted probability')
                xlabel('trial')
            end
            if obj.plot_state_given_cstar3
                figure
                plot(P.state_given_cstar3,'Color','k','LineWidth',line_width)
                axis([0 T -1.5 1.5])
                set(gca,'YTick',[-1 0 1],'XTick',[0 T],'FontSize',font_size)
                ylabel('E[ state | c^*]')
                title('c^* is the context with the highest predicted probability')
                xlabel('trial')
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
                C.cues = copper(max(obj.cues));
            end
            
        end
        
    end
end