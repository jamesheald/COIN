classdef COIN

    properties
        
        % model implementation
        P = 100                                   % number of particles
        R = 1                                     % number of runs to perform, each conditioned on a different observation noise sequence
        maxC = 10                                 % maximum number of contexts that the model can instantiate
        maxCores = 0                              % maximum number of CPU cores available (0 implements serial processing)
        
        % parameters
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
        trials                                    % trials to simulate (set to 1:T later if left empty)
        observeY                                  % is the state feedback observed or not on a trial (set later based on x)?
        eraserTrials                              % trials on which to overwrite context probabilities with the stationary probabilities
        
        % measured adaptation data
        adaptation                                % vector of adaptation data (NaN if adaptation not measured on a trial)
        
        % store
        store = {}                                % variables to store in memory
        
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

    end

    methods
        
        function [S,w] = run_COIN(obj)
            
            % set properties based on the vaues of other properties
            obj = set_property_values(obj);
            
            % preallocate memory
            S = cell(1,obj.R);
            
            if isempty(obj.adaptation)
                
                % perform obj.R 'open-loop' runs. each run is associated
                % with a different observation noise sequence.
                
                parfor (run = 1:obj.R,obj.maxCores)
                    S{run} = obj.main_loop.stored;
                end
                
                % assign equal weights to all runs
                w = ones(1,obj.R)/obj.R;

            else
                
                % perform obj.R 'closed-loop' run by (optionally)
                % resampling runs whenever adaptation is measured.
                % in between measurements, the simulations are open-loop.
                % each run is associated with a different observation
                % noise sequence.

                % preallocate memory
                D_in = cell(1,obj.R);
                D_out = cell(1,obj.R);
                
                % initialise weights to be uniform
                w = ones(1,obj.R)/obj.R;

                % effective sample size threshold for resampling
                thresholdESS = 0.5*obj.R;

                % trials on which adaptation was measured
                aT = find(~isnan(obj.adaptation));

                % simulate trials inbetween trials on which adaptation was measured
                for i = 1:numel(aT)

                    if i == 1
                        obj.trials = 1:aT(i);
                        fprintf('running the COIN model from trial 1 to trial %d\n',aT(i))
                    else
                        obj.trials = aT(i-1)+1:aT(i);
                        fprintf('running the COIN model from trial %d to trial %d\n',aT(i-1)+1,aT(i))
                    end

                    parfor (run = 1:obj.R,obj.maxCores)
                        if i == 1
                            D_out{run} = obj.main_loop;
                        else
                            D_out{run} = obj.main_loop(D_in{run});
                        end
                    end

                    % calculate the log likelihood
                    logLikelihood = zeros(1,obj.R);
                    for run = 1:obj.R
                        error = D_out{run}.stored.yHat(aT(i)) - obj.adaptation(aT(i));
                        logLikelihood(run) = -(log(2*pi*obj.sigmaM^2) + (error/obj.sigmaM).^2)/2; 
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
                        for run = 1:obj.R
                            D_in{run} = D_out{iResamp(run)};
                        end
                        w = ones(1,obj.R)/obj.R;
                    else
                        fprintf('effective sample size = %.1f --- not resampling\n',ESS)
                        D_in = D_out;
                    end

                end
                
                if aT(end) == obj.T
                    
                    for run = 1:obj.R
                        S{run} = D_in{run}.stored;
                    end

                elseif aT(end) < obj.T

                    % simulate to the last trial
                    fprintf('running the COIN model from trial %d to trial %d\n',aT(end)+1,obj.T)

                    obj.trials = aT(end)+1:obj.T;

                    parfor (run = 1:obj.R,obj.maxCores)
                        S{run} = main_loop(D_in{run}).stored;
                    end
                    
                end

            end
            
        end
        
        function obj = set_property_values(obj)
            
            % number of trials
            obj.T = numel(obj.x);
            
            % trials to simulate
            if isempty(obj.trials)
                obj.trials = 1:obj.T;
            end
            
            % treat channel trials as non observations with respect to the state feedback
            obj.observeY = ones(1,obj.T);
            obj.observeY(isnan(obj.x)) = 0;
            
            % ensure adMu is a column vector
            if isrow(obj.adMu)
                obj.adMu = obj.adMu';
            end
            
            % self-transition bias
            obj.kappa = obj.alpha*obj.rho/(1-obj.rho);
            
            % observation noise variance
            obj.sigmaR = sqrt(obj.sigmaS^2 + obj.sigmaM^2);
            
            % matrix of context-dependent observation vectors
            obj.H = eye(obj.maxC+1);
            
            if isempty(obj.q) 
                obj.cuesExist = 0;
            else
                obj.cuesExist = 1;
            end
            
            % specify variables that need to be stored for plots
            tmp = {};
            if obj.xPredPlot
                if isempty(obj.gridX)
                    error('Specify points at which to evaluate the state | context distribution. Use property ''gridX''.')
                end
                tmp = cat(2,tmp,{'xPred','vPred'});
            end
            if obj.cPredPlot
                tmp = cat(2,tmp,'cPred');
            end
            if obj.cFiltPlot
                tmp = cat(2,tmp,'cFilt');
            end
            if obj.cInfPlot
                tmp = cat(2,tmp,'cInf');
            end
            if obj.adPlot
                if isempty(obj.gridA)
                    error('Specify points at which to evaluate the retention | context distribution. Use property ''gridA''.')
                end
                if isempty(obj.gridD)
                    error('Specify points at which to evaluate the drift | context distribution. Use property ''gridD''.')
                end
                tmp = cat(2,tmp,{'adMu','adCovar'});
            end
            if obj.bPlot
                if isempty(obj.gridB)
                    error('Specify points at which to evaluate the bias | context distribution. Use property ''gridB''.')
                end
                if obj.learnB
                    tmp = cat(2,tmp,{'bMu','bVar'});
                else
                    error('You must learn the measurement bias parameter to use bPlot. Set property ''learnB'' to true.')
                end
            end
            if obj.transitionMatrixPlot
                tmp = cat(2,tmp,'transitionMatrix');
            end
            if obj.emissionMatrixPlot
                if obj.cuesExist
                    tmp = cat(2,tmp,'emissionMatrix');
                else
                    error('An experiment must have sensory cues to use emissionMatrixPlot.')
                end
            end
            if obj.xPredMargPlot
                if isempty(obj.gridX)
                    error('Specify points at which to evaluate the state distribution. Use property ''gridX''.')
                end
                tmp = cat(2,tmp,'xPredMarg');
            end
            if obj.bPredMargPlot
                if isempty(obj.gridB)
                    error('Specify points at which to evaluate the bias distribution. Use property ''gridB''.')
                end
                if obj.learnB
                    tmp = cat(2,tmp,'bPredMarg');
                else
                    error('You must learn the measurement bias parameter to use bPredMargPlot. Set property ''learnB'' to true.')
                end
            end
            if obj.yPredMargPlot
                if isempty(obj.gridY)
                    error('Specify points at which to evaluate the state feedback distribution. Use property ''gridY''.')
                end
                tmp = cat(2,tmp,'yPredMarg');
            end
            if obj.xHatPlot
                tmp = cat(2,tmp,'xHat');
            end
            if obj.explicitPlot
                tmp = cat(2,tmp,'explicit');
            end
            if obj.implicitPlot
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
            
            if ~isempty(obj.adaptation) && numel(obj.adaptation) ~= obj.T
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

            logLikelihood = zeros(obj(1).R,1);
            parfor (run = 1:obj(1).R,obj(1).maxCores)

                model = zeros(n,S);
                for s = 1:S

                    % model adaptation
                    model(:,s) = obj(s).main_loop.stored.yHat(aT(:,s));
                
                end

                % error between average model adaptation and average measured adaptation
                error = mean(model-data,2);

                % log likelihood (probability of data given parameters)
                logLikelihood(run) = sum(-(log(2*pi*obj(1).sigmaM^2/S) + error.^2/(obj(1).sigmaM.^2/S))/2); % variance scaled by the number of participants

            end

            % negative of the log of the average likelihood across runs
            objective = -(log(1/obj(1).R) + obj(1).log_sum_exp(logLikelihood));
            
%             obj = set_property_values(obj);
%             
%             if isrow(obj.adaptation)
%                 obj.adaptation = obj.adaptation';
%             end
%             
%             % trials on which adaptation was measured
%             aT = logical(~isnan(obj.adaptation));
%             
%             % measured adaptation
%             data = obj.adaptation(aT);
% 
%             logLikelihood = zeros(obj.R,1);
%             parfor (run = 1:obj.R,obj.maxCores)
%                 
%                 % model adaptation
%                 model = obj.main_loop.stored.yHat(aT);
%                 
%                 % error between model and measured adaptation
%                 error = model - data;
%                 
%                 % log likelihood (probability of data given parameters)
%                 logLikelihood(run) = sum(-(log(2*pi*obj.sigmaM^2) + (error/obj.sigmaM).^2)/2);
% 
%             end
% 
%             % negative of the log of the average likelihood across runs
%             objectiveEst = -(log(1/obj.R) + obj.log_sum_exp(logLikelihood));
        
        end

        
        function D = main_loop(obj,varargin)

            if obj.trials(1) == 1
                D = obj.initialise_COIN;                                % initialise the model
            else
                D = varargin{1};
            end
            for trial = obj.trials
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
            
            % current trial
            D.t = 0;

            % number of contexts instantiated so far
            D.C = zeros(1,obj.P);

            % context transition counts
            D.n = zeros(obj.maxC+1,obj.maxC+1,obj.P);

            % context
            D.c = ones(1,obj.P); % treat trial 1 as a (context 1) self transition
                
            if obj.cuesExist
                
                D = obj.check_cue_labels(D);
                
                % number of contextual cues observed so far
                D.Q = 0;
                
                % cue emission counts
                D.nE = zeros(obj.maxC+1,max(obj.q)+1,obj.P);
            end

            % sufficient statistics for the parameters of the state transition
            % function
            D.adSS1 = zeros(obj.maxC+1,obj.P,2);
            D.adSS2 = zeros(obj.maxC+1,obj.P,2,2);

            % sufficient statistics for the parameters of the observation function
            D.bSS1 = zeros(obj.maxC+1,obj.P);
            D.bSS2 = zeros(obj.maxC+1,obj.P);

            % sample parameters from the prior
            D = sample_parameters(obj,D);
            
            % mean and variance of state (stationary distribution)
            D.xFilt = D.d./(1-D.a);
            D.vFilt = obj.sigmaQ^2./(1-D.a.^2);
            
        end
        
        function D = predict_context(obj,D)
            
            if ismember(D.t,obj.eraserTrials)
                % if some event (e.g. a working memory task) causes the context 
                % probabilities to be erased, set them to their stationary values
                for particle = 1:obj.P
                    C = sum(D.transitionMatrix(:,1,particle)>0);
                    T = D.transitionMatrix(1:C,1:C,particle);        
                    D.cPrior(1:C,particle) = obj.stationary_distribution(T);
                end
            else
                i = sub2ind(size(D.transitionMatrix),repmat(D.c,[obj.maxC+1,1]),repmat(1:obj.maxC+1,[obj.P,1])',repmat(1:obj.P,[obj.maxC+1,1]));
                D.cPrior = D.transitionMatrix(i);
            end

            if obj.cuesExist
                i = sub2ind(size(D.emissionMatrix),repmat(1:obj.maxC+1,[obj.P,1])',repmat(obj.q(D.t),[obj.maxC+1,obj.P]),repmat(1:obj.P,[obj.maxC+1,1]));
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
            D.vPred = D.a.*D.vFilt.*D.a + obj.sigmaQ^2;

            % index of novel states
            iNewX = sub2ind([obj.maxC+1,obj.P],D.C+1,1:obj.P);

            % novel states are distributed according to the stationary distribution
            D.xPred(iNewX) = D.d(iNewX)./(1-D.a(iNewX));
            D.vPred(iNewX) = obj.sigmaQ^2./(1-D.a(iNewX).^2);

            % predict state (marginalise over contexts and particles)
            % mean of distribution
            D.xHat = sum(D.cPred.*D.xPred,'all')/obj.P;

            if any(strcmp(obj.store,'explicit'))
                if D.t == 1
                    i = ones(1,obj.P);
                else
                    [~,i] = max(D.cFilt,[],1);
                end
                i = sub2ind(size(D.xPred),i,1:obj.P);
                D.explicit = mean(D.xPred(i));
            end

        end
        
        function D = predict_state_feedback(obj,D)  

            % predict state feedback for each context
            D.yPred = D.xPred + D.b;

            % variance of state feedback prediction for each context
            D.pPred = D.vPred + obj.sigmaR^2;
            
            D = obj.compute_marginal_distribution(D);

            % predict state feedback (marginalise over contexts and particles)
            % mean of distribution
            D.yHat = sum(D.cPred.*D.yPred,'all')/obj.P;

            if any(strcmp(obj.store,'implicit'))
                D.implicit = D.yHat - D.xHat;
            end

            % sensory and motor noise
            D.sensoryNoise = obj.sigmaS*randn;
            D.motorNoise = obj.sigmaM*randn;

            % state feedback
            D.y = obj.x(D.t) + D.sensoryNoise + D.motorNoise;

            % state feedback prediction error
            D.e = D.y - D.yPred;

        end
        
        function D = resample_particles(obj,D)

            D.pY = normpdf(D.y,D.yPred,sqrt(D.pPred)); % p(y_t|c_t)

            if obj.observeY(D.t)
                if obj.cuesExist
                    pC = log(D.cPrior) + log(D.pQ) + log(D.pY); % log p(y_t,q_t,c_t)
                else
                    pC = log(D.cPrior) + log(D.pY); % log p(y_t,c_t)
                end
            else
                if obj.cuesExist
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
            if obj.observeY(D.t) || obj.cuesExist
                D.iResamp = obj.systematic_resampling(w);
            else
                D.iResamp = 1:obj.P;
            end
            
            % optionally store predictive variable
            % n.b. these are stored before resampling (so that they do not condition on the current observations)
            for i = 1:numel(obj.store)
                if contains(obj.store{i},'Pred')
                    if D.t == 1
                        % preallocate memory for variables to be stored
                        s = [eval(sprintf('size(D.%s)',obj.store{i})) obj.T];
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

            if obj.cuesExist 
                D.betaE  = D.betaE(:,D.iResamp);
                D.nE = D.nE(:,:,D.iResamp);
            end

            D.a = D.a(:,D.iResamp);
            D.d = D.d(:,D.iResamp);

            D.adSS1 = D.adSS1(:,D.iResamp,:);
            D.adSS2 = D.adSS2(:,D.iResamp,:,:);

            if obj.learnB
                D.bSS1 = D.bSS1(:,D.iResamp);
                D.bSS2 = D.bSS2(:,D.iResamp);
            end

        end

        function D = sample_context(obj,D)

            % sample the context
            D.c = sum(rand(1,obj.P) > cumsum(D.cFilt),1) + 1;

            % incremement the context count
            D.pNewX = find(D.c > D.C);
            D.pOldX = find(D.c <= D.C);
            D.C(D.pNewX) = D.C(D.pNewX) + 1;

            pBetaX = D.pNewX(D.C(D.pNewX) ~= obj.maxC);
            iBu = sub2ind([obj.maxC+1,obj.P],D.c(pBetaX),pBetaX);

            % sample the next stick-breaking weight
            beta = betarnd(1,obj.gamma*ones(1,numel(pBetaX)));

            % update the global transition distribution
            D.beta(iBu+1) = D.beta(iBu).*(1-beta);
            D.beta(iBu) = D.beta(iBu).*beta;

            if obj.cuesExist
                if obj.q(D.t) > D.Q 
                    % increment the cue count
                    D.Q = D.Q + 1;

                    % sample the next stick-breaking weight
                    beta = betarnd(1,obj.gammaE*ones(1,obj.P));

                    % update the global emission distribution
                    D.betaE(D.Q+1,:) = D.betaE(D.Q,:).*(1-beta);
                    D.betaE(D.Q,:) = D.betaE(D.Q,:).*beta;
                end

            end

        end

        function D = update_belief_about_states(obj,D)

            D.k = D.vPred./D.pPred;
            if obj.observeY(D.t)
                D.xFilt = D.xPred + D.k.*D.e.*obj.H(D.c,:)';
                D.vFilt = (1 - D.k.*obj.H(D.c,:)').*D.vPred;
            else
                D.xFilt = D.xPred;
                D.vFilt = D.vPred;
            end

        end
        
        function D = sample_states(obj,D)

            nNewX = numel(D.pNewX);
            iOldX = sub2ind([obj.maxC+1,obj.P],D.c(D.pOldX),D.pOldX);
            iNewX = sub2ind([obj.maxC+1,obj.P],D.c(D.pNewX),D.pNewX);

            % for states that have been observed before, sample x_{t-1}, and then sample x_{t} given x_{t-1}

                % sample x_{t-1} using a fixed-lag (lag 1) forward-backward smoother
                g = D.a.*D.vPrev./D.vPred;
                m = D.xPrev + g.*(D.xFilt - D.xPred);
                v = D.vPrev + g.*(D.vFilt - D.vPred).*g;
                D.xPrevSamp = m + sqrt(v).*randn(obj.maxC+1,obj.P);

                % sample x_t conditioned on x_{t-1} and y_t
                if obj.observeY(D.t)
                    w = (D.a.*D.xPrevSamp + D.d)./obj.sigmaQ^2 + obj.H(D.c,:)'./obj.sigmaR^2.*(D.y - D.b);
                    v = 1./(1./obj.sigmaQ^2 + obj.H(D.c,:)'./obj.sigmaR^2);
                else
                    w = (D.a.*D.xPrevSamp + D.d)./obj.sigmaQ^2;
                    v = 1./(1./obj.sigmaQ^2);
                end
                D.xSampOld = v.*w + sqrt(v).*randn(obj.maxC+1,obj.P);

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
            % cue emission probabilities
            if obj.cuesExist 
                D = obj.update_sufficient_statistics_phi(D);
            end

            if D.t > 1
                % update the sufficient statistics for the parameters of the 
                % state transition function
                D = obj.update_sufficient_statistics_ad(D);
            end

            % update the sufficient statistics for the parameters of the 
            % observation function
            if obj.learnB && obj.observeY(D.t)
                D = obj.update_sufficient_statistics_b(D);
            end

        end
         
        function D = sample_parameters(obj,D)

            % sample beta
            D = obj.sample_beta(D);

            % update context transition matrix
            D = obj.update_transition_matrix(D);

            if obj.cuesExist 
                % sample betaE
                D = obj.sample_betaE(D);

                % update cue emission matrix
                D = obj.update_emission_matrix(D);
            end

            % sample the parameters of the state transition function
            D = obj.sample_ad(D);

            % sample the parameters of the observation function
            if obj.learnB
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
                        s = [eval(sprintf('size(D.%s)',obj.store{i})) obj.T];
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
            xa = ones(obj.maxC+1,obj.P,2);
            xa(:,:,1) = D.xPrevSamp;

            % identify states that are not novel
            I = reshape(sum(D.n,2),[obj.maxC+1,obj.P]) > 0;

            SS = D.xSampOld.*xa; % x_t*[x_{t-1}; 1]
            D.adSS1 = D.adSS1 + SS.*I;

            SS = reshape(xa,[obj.maxC+1,obj.P,2]).*reshape(xa,[obj.maxC+1,obj.P,1,2]); % [x_{t-1}; 1]*[x_{t-1}; 1]'
            D.adSS2 = D.adSS2 + SS.*I;

        end
         
        function D = update_sufficient_statistics_b(obj,D)

            SS = D.y - D.xSamp; % y_t - x_t
            D.bSS1(D.iX) = D.bSS1(D.iX) + SS;
            D.bSS2(D.iX) = D.bSS2(D.iX) + 1; % 1(c_t = j)
            
        end

        function D = update_sufficient_statistics_phi(obj,D)

            i = sub2ind([obj.maxC+1,max(obj.q)+1,obj.P],D.c,obj.q(D.t)*ones(1,obj.P),1:obj.P); % 1(c_t = j, q_t = k)
            D.nE(i) = D.nE(i) + 1;   

        end
         
        function D = update_sufficient_statistics_pi(obj,D)

            i = sub2ind([obj.maxC+1,obj.maxC+1,obj.P],D.cPrev,D.c,1:obj.P); % 1(c_{t-1} = i, c_t = j)
            D.n(i) = D.n(i) + 1;
            
        end
         
        function D = sample_ad(obj,D)

            % update the parameters of the posterior
            D.adCovar = obj.per_slice_invert(obj.adLambda + permute(reshape(D.adSS2,[(obj.maxC+1)*obj.P,2,2]),[2,3,1])/obj.sigmaQ^2);
            D.adMu = obj.per_slice_multiply(D.adCovar,obj.adLambda*obj.adMu + reshape(D.adSS1,[(obj.maxC+1)*obj.P,2])'/obj.sigmaQ^2);
            
            % sample the parameters of the state transition function
            ad = obj.sample_from_truncated_bivariate_normal(D.adMu,D.adCovar);
            D.a = reshape(ad(1,:),[obj.maxC+1,obj.P]);
            D.d = reshape(ad(2,:),[obj.maxC+1,obj.P]);
            
            % reshape
            D.adMu = reshape(D.adMu,[2,obj.maxC+1,obj.P]);
            D.adCovar = reshape(D.adCovar,[2,2,obj.maxC+1,obj.P]);

        end
         
        function D = sample_b(obj,D)

            % update the parameters of the posterior
            D.bVar = 1./(obj.bLambda + D.bSS2./obj.sigmaR^2);
            D.bMu = D.bVar.*(obj.bLambda.*obj.bMu + D.bSS1./obj.sigmaR^2);

            % sample the parameters of the observation function
            D.b = obj.sample_from_univariate_normal(D.bMu,D.bVar);

        end
         
        function D = sample_beta(obj,D)

            if D.t == 0
                % global transition distribution
                D.beta = zeros(obj.maxC+1,obj.P);
                D.beta(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.m = randnumtable(permute(obj.alpha*D.beta,[3,1,2]) + obj.kappa*eye(obj.maxC+1),D.n);

                % sample the number of tables in restaurant i considering dish j
                barM = D.m;
                if obj.rho > 0
                    i = sub2ind([obj.maxC+1,obj.maxC+1,obj.P],repmat(1:obj.maxC+1,[obj.P,1])',repmat(1:obj.maxC+1,[obj.P,1])',repmat(1:obj.P,[obj.maxC+1,1]));
                    p = obj.rho./(obj.rho + D.beta(D.m(i) ~= 0)*(1-obj.rho));
                    i = i(D.m(i) ~= 0);
                    barM(i) = D.m(i) - randbinom(p,D.m(i));
                end
                barM(1,1,(barM(1,1,:) == 0)) = 1;

                % sample beta
                i = find(D.C ~= obj.maxC);
                iBu = sub2ind([obj.maxC+1,obj.P],D.C(i)+1,i);
                barMGam = squeeze(sum(barM,1));
                barMGam(iBu) = obj.gamma;
                D.beta = obj.sample_from_dirichlet(barMGam);
            end

        end
         
        function D = sample_betaE(obj,D)

            if D.t == 0
                % global cue emission distribution
                D.betaE = zeros(max(obj.q)+1,obj.P);
                D.betaE(1,:) = 1;
            else
                % sample the number of tables in restaurant i serving dish j
                D.mE = randnumtable(repmat(permute(obj.alphaE.*D.betaE,[3,1,2]),[obj.maxC+1,1,1]),D.nE);

                % sample betaE
                mGam = reshape(sum(D.mE,1),[max(obj.q)+1,obj.P]);
                mGam(D.Q+1,:) = obj.gammaE;
                D.betaE = obj.sample_from_dirichlet(mGam);
            end

        end
         
        function D = update_emission_matrix(obj,D)

            D.emissionMatrix = reshape(obj.alphaE.*D.betaE,[1,max(obj.q)+1,obj.P]) + D.nE;
            D.emissionMatrix = D.emissionMatrix./sum(D.emissionMatrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.beta,[obj.maxC+1,1,obj.P]) > 0;
            D.emissionMatrix = D.emissionMatrix.*I;

        end
        
        function D = update_transition_matrix(obj,D)

            D.transitionMatrix = reshape(obj.alpha*D.beta,[1,obj.maxC+1,obj.P]) + D.n + obj.kappa*eye(obj.maxC+1);
            D.transitionMatrix = D.transitionMatrix./sum(D.transitionMatrix,2);

            % remove contexts with zero mass under the global transition distribution
            I = reshape(D.beta,[obj.maxC+1,1,obj.P]) > 0;
            D.transitionMatrix = D.transitionMatrix.*I;

            % compute stationary context probabilities if required
            if any(strcmp(obj.store,'cInf')) && D.t > 0
                D.cInf = zeros(obj.maxC+1,obj.P);
                for particle = 1:obj.P
                    C = D.C(particle);
                    T = D.transitionMatrix(1:C+1,1:C+1,particle);
                    D.cInf(1:C+1,particle) = obj.stationary_distribution(T);
                end
            end

        end
         
        function D = check_cue_labels(obj,D)

            % check cues are numbered according to the order they were presented in the experiment
            for trial = 1:obj.T
                if trial == 1
                    if ~eq(obj.q(trial),1)
                        D = obj.renumber_cues(D);
                        break
                    end
                else
                    if ~ismember(obj.q(trial),obj.q(1:trial-1)) && ~eq(obj.q(trial),max(obj.q(1:trial-1))+1)
                        D = obj.renumber_cues(D);
                        break
                    end
                end
            end

            function D = renumber_cues(obj,D)

                cueOrder = unique(obj.q,'stable');
                if isrow(obj.q)
                    [obj.q,~] = find(eq(obj.q,cueOrder'));
                else
                    [obj.q,~] = find(eq(obj.q,cueOrder')');
                end
                fprintf('Cues have been renumbered according to the order they were presented in the experiment.\n')
                
            end
            
        end
         
        function D = compute_marginal_distribution(obj,D)
             
            if any(strcmp(obj.store,'xPredMarg'))
                % predict state (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.gridX,[1,1,numel(obj.gridX)]);
                mu = D.xPred;
                sd = sqrt(D.vPred);
                D.xPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.P;
            end
            if any(strcmp(obj.store,'bPredMarg'))
                % predict bias (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.gridB,[1,1,numel(obj.gridB)]);
                mu = D.bMu;
                sd = sqrt(D.bVar);
                D.bPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.P;
            end
            if any(strcmp(obj.store,'yPredMarg'))
                % predict state feedback (marginalise over contexts and particles)
                % entire distribution (discretised)
                x = reshape(obj.gridY,[1,1,numel(obj.gridY)]);
                mu = D.yPred;
                sd = sqrt(D.pPred);
                D.yPredMarg = sum(D.cPred.*normpdf(x,mu,sd),[1,2])/obj.P;
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

            % truncation bounds for the state transition coefficient
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

            x = mu + sqrt(sigma).*randn(obj.maxC+1,obj.P);

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
        
        function P = plot_COIN(obj,S,runWeight)
            
            % set properties based on the vaues of other properties
            obj = set_property_values(obj);
            
            [P,S,optAssignment,from_unique,cSeq,C] = obj.find_optimal_context_labels(S,runWeight);
            
            [P,~] = obj.compute_variables_for_plotting(P,S,optAssignment,from_unique,cSeq,C,runWeight);
            
            obj.generate_figures(P);
            
        end
        
        function [P,S,optAssignment,from_unique,cSeq,C] = find_optimal_context_labels(obj,S,runWeight)
            
            iResamp = obj.resample_indices(S);
            
            cSeq = obj.context_sequences(S,iResamp);

            [C,~,~,P.posteriorMode] = obj.posterior_number_of_contexts(cSeq,runWeight);
            
            % context label permutations
            L = flipud(perms(1:max(P.posteriorMode)));
            L = permute(L,[2,3,1]);
            nPerm = factorial(max(P.posteriorMode));
            
            % preallocate memory
            f = cell(1,obj.T);
            to_unique = cell(1,obj.T);
            from_unique = cell(1,obj.T);
            optAssignment = cell(1,obj.T);
            for trial = 1:obj.T

                if ~mod(trial,50)
                    fprintf('finding typical context sequence and optimal context labels, trial = %d\n',trial)
                end

                    % exclude sequences for which C > max(P.posteriorMode) as 
                    % these sequences (and their descendents) will never be 
                    % analysed
                    f{trial} = find(C(:,trial) <= max(P.posteriorMode));

                    % identify unique sequences (to avoid performing the
                    % same computations multiple times)
                    [uniqueSequences,to_unique{trial},from_unique{trial}] = unique(cSeq{trial}(f{trial},:),'rows');

                    % number of unique sequences
                    nSeq = size(uniqueSequences,1);

                    % identify particles that have the same number of
                    % contexts as the most common number of contexts (only
                    % these particles will be analysed)
                    idx = logical(C(f{trial},trial) == P.posteriorMode(trial));

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
                    
                    if numel(unique(runWeight)) > 1
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
                    optAssignment{trial} = permute(L(1:P.posteriorMode(trial),:,j),[3,1,2]);

            end

        end
        
        function iResamp = resample_indices(obj,S)
            
            iResamp = zeros(obj.P*obj.R,obj.T);
            for run = 1:obj.R
                p = obj.P*(run-1) + (1:obj.P);
                iResamp(p,:) = obj.P*(run-1) + S{run}.iResamp;
            end
            
        end
        
        function cSeq = context_sequences(obj,S,iResamp)
            
            cSeq = cell(1,obj.T);
            for run = 1:obj.R
                p = obj.P*(run-1) + (1:obj.P);
                for trial = 1:obj.T
                    if run == 1
                        cSeq{trial} = zeros(obj.P*obj.R,trial);
                    end
                    if trial > 1
                        cSeq{trial}(p,1:trial-1) = cSeq{trial-1}(p,:);
                        cSeq{trial}(p,:) = cSeq{trial}(iResamp(p,trial),:);
                    end
                    cSeq{trial}(p,trial) = S{run}.c(:,trial);
                end
            end
            
        end
        
        function [C,posterior,posteriorMean,posteriorMode] = posterior_number_of_contexts(obj,cSeq,runWeight)
            
            % number of contexts
            C = zeros(obj.P*obj.R,obj.T);
            for run = 1:obj.R
                p = obj.P*(run-1) + (1:obj.P);
                for trial = 1:obj.T
                    C(p,trial) = max(cSeq{trial}(p,:),[],2);
                end
            end

            runWeight = reshape(runWeight,[obj.R,1]);
            particleWeight = repelem(runWeight,obj.P)/obj.P;

            posterior = zeros(obj.maxC+1,obj.T);
            posteriorMean = zeros(1,obj.T);
            posteriorMode = zeros(1,obj.T);
            for time = 1:obj.T

                for context = 1:max(C(:,time),[],'all')
                    posterior(context,time) = sum((C(:,time) == context).*particleWeight,'all');
                end
                posteriorMean(time) = (1:obj.maxC+1)*posterior(:,time);
                [~,posteriorMode(time)] = max(posterior(:,time));

            end
            
        end
        
        function [P,S] = compute_variables_for_plotting(obj,P,S,optAssignment,from_unique,cSeq,C,runWeight)

            P = obj.preallocate_memory_for_plot_variables(P);
            
            nParticlesUsed = zeros(obj.T,obj.R);
            for trial = 1:obj.T
                if ~mod(trial,50)
                    fprintf('permuting context labels, trial = %d\n',trial)
                end
                
                % cumulative number of particles for which C <= max(P.posteriorMode)
                N = 0;

                for run = 1:obj.R
                    % indices of particles of the current run
                    p = obj.P*(run-1) + (1:obj.P);
                    
                    % indices of particles that are either valid now or 
                    % could be valid in the future: C <= max(P.posteriorMode)
                    validFuture = find(C(p,trial) <= max(P.posteriorMode));
                    
                    % indices of particles that are valid now: C == P.posteriorMode(trial)
                    validNow = find(C(p,trial) == P.posteriorMode(trial))';
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

            P = obj.integrate_over_runs(P,S,runWeight);

            P = obj.normalise_relabelled_variables(P,nParticlesUsed,runWeight);

            % the predicted state distribution for a novel context is the marginal
            % stationary distribution of the state after integrating out 
            % the drift and retention parameters under the prior
            P.xPred(:,:,end) = repmat(nanmean(P.xPred(:,:,end),2),[1,obj.T]);
            
            % (weighted) average number of particles used per run in this analysis
            P.nParticles = obj.weighted_sum_along_dimension(nParticlesUsed,runWeight,2);

        end

        function S = relabel_context_variables(obj,S,optAssignment,novelContext,particle,trial,run)

            C = numel(optAssignment);

            % predictive distributions
            if trial < obj.T
                if obj.xPredPlot
                    S{run}.xPred(optAssignment,particle,trial+1) = S{run}.xPred(1:C,particle,trial+1);
                    S{run}.vPred(optAssignment,particle,trial+1) = S{run}.vPred(1:C,particle,trial+1);
                end
                if obj.cPredPlot
                    S{run}.cPred(optAssignment,particle,trial+1) = S{run}.cPred(1:C,particle,trial+1);
                end
            end

            if obj.cFiltPlot
                if trial == 1 || novelContext 
                    optAssignment2 = [optAssignment(1:end-1) C+1 optAssignment(end)];
                else
                    optAssignment2 = [optAssignment C+1];
                end
                S{run}.cFilt(1:C+1,particle,trial) = S{run}.cFilt(optAssignment2,particle,trial);
            end
            if obj.cInfPlot
                S{run}.cInf(optAssignment,particle,trial) = S{run}.cInf(1:C,particle,trial);
            end
            if obj.adPlot
                S{run}.adMu(:,optAssignment,particle,trial) = S{run}.adMu(:,1:C,particle,trial);
                S{run}.adCovar(:,:,optAssignment,particle,trial) = S{run}.adCovar(:,:,1:C,particle,trial);
            end
            if obj.bPlot
                S{run}.bMu(optAssignment,particle,trial) = S{run}.bMu(1:C,particle,trial);
                S{run}.bVar(optAssignment,particle,trial) = S{run}.bVar(1:C,particle,trial);
            end
            if obj.emissionMatrixPlot
                S{run}.emissionMatrix(optAssignment,:,particle,trial) = S{run}.emissionMatrix(1:C,:,particle,trial);
            end
            if obj.transitionMatrixPlot
                S{run}.transitionMatrix(1:C,1:C+1,particle,trial) = ...
                obj.permute_transition_matrix_columns_and_rows(S{run}.transitionMatrix(1:C,1:C+1,particle,trial),optAssignment);
            end

        end
        
        function P = permute_transition_matrix_columns_and_rows(obj,T,optAssignment)

                C = numel(optAssignment);

                [i_map,~] = find(optAssignment' == 1:C); % inverse mapping

                i = sub2ind(size(T),repmat(i_map,[1,C+1]),repmat([i_map' C+1],[C,1]));
                P = T(i);

        end
        
        function P = preallocate_memory_for_plot_variables(obj,P)

            if obj.xPredPlot
                P.xPred = NaN(numel(obj.gridX),obj.T,max(P.posteriorMode)+1,obj.R);
            end
            if obj.cPredPlot
                P.cPred = NaN(obj.T,max(P.posteriorMode)+1,obj.R);
                P.cPred(1,end,:) = 1;
            end
            if obj.xPredPlot && obj.cPredPlot
                P.xHatReduced = NaN(obj.T,obj.R);
            end
            if obj.cFiltPlot
                P.cFilt = NaN(obj.T,max(P.posteriorMode)+1,obj.R);
            end
            if obj.cInfPlot
                P.cInf = NaN(obj.T,max(P.posteriorMode)+1,obj.R);
            end
            if obj.adPlot
                P.retention = zeros(numel(obj.gridA),obj.T,max(P.posteriorMode),obj.R);
                P.drift = zeros(numel(obj.gridD),obj.T,max(P.posteriorMode),obj.R);
            end
            if obj.bPlot
                P.bias = zeros(numel(obj.gridB),obj.T,max(P.posteriorMode),obj.R);
            end
            if obj.emissionMatrixPlot
                P.emissionMatrix = NaN(max(P.posteriorMode),max(obj.q)+1,obj.T,obj.R);
            end
            if obj.transitionMatrixPlot
                P.transitionMatrix = NaN(max(P.posteriorMode),max(P.posteriorMode)+1,obj.T,obj.R);
            end
            if obj.xPredMargPlot
                P.xPredMarg = NaN(numel(obj.gridX),obj.T,obj.R);
            end
            if obj.bPredMargPlot
                P.bPredMarg = NaN(numel(obj.gridB),obj.T,obj.R);
            end
            if obj.yPredMargPlot
                P.yPredMarg = NaN(numel(obj.gridY),obj.T,obj.R);
            end
            if obj.explicitPlot
                P.explicit = NaN(obj.T,obj.R);
            end
            if obj.implicitPlot
                P.implicit = NaN(obj.T,obj.R);
            end
            if obj.xHatPlot
                P.xHat = NaN(obj.T,obj.R);
            end
            P.yHat = NaN(obj.T,obj.R);  

        end
        
        function P = integrate_over_particles(obj,S,P,particles,trial,run)

            C = P.posteriorMode(trial);
            novelContext = max(P.posteriorMode)+1;

            % predictive distributions
            if trial < obj.T
                if obj.xPredPlot
                    mu = permute(S{run}.xPred(1:C+1,particles,trial+1),[3,2,1]);
                    sd = permute(sqrt(S{run}.vPred(1:C+1,particles,trial+1)),[3,2,1]);
                    P.xPred(:,trial+1,[1:C novelContext],run) = sum(normpdf(obj.gridX',mu,sd),2);
                end
                if obj.cPredPlot
                    P.cPred(trial+1,[1:C novelContext],run) = sum(S{run}.cPred(1:C+1,particles,trial+1),2);
                end
                if obj.xPredPlot && obj.cPredPlot
                    P.xHatReduced(trial+1,run) = sum(S{run}.cPred(:,particles,trial+1).*S{run}.xPred(:,particles,trial+1),'all');
                end
            end

            if obj.cFiltPlot
                P.cFilt(trial,[1:C novelContext],run) = sum(S{run}.cFilt(1:C+1,particles,trial),2);
            end
            if obj.cInfPlot
                P.cInf(trial,[1:C novelContext],run) = sum(S{run}.cInf(1:C+1,particles,trial),2);
            end
            if obj.adPlot        
                mu = permute(S{run}.adMu(1,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S{run}.adCovar(1,1,1:C,particles,trial)),[1,4,3,2]);
                P.retention(:,trial,1:C,run) = sum(normpdf(obj.gridA',mu,sd),2);

                mu = permute(S{run}.adMu(2,1:C,particles,trial),[1,3,2]);
                sd = permute(sqrt(S{run}.adCovar(2,2,1:C,particles,trial)),[1,4,3,2]);
                P.drift(:,trial,1:C,run) = sum(normpdf(obj.gridD',mu,sd),2);  
            end
            if obj.bPlot
                mu = permute(S{run}.bMu(1:C,particles,trial),[3,2,1]);
                sd = permute(sqrt(S{run}.bVar(1:C,particles,trial)),[3,2,1]);
                P.bias(:,trial,1:C,run) = sum(normpdf(obj.gridB',mu,sd),2);  
            end
            if obj.emissionMatrixPlot
                P.emissionMatrix(1:C,[1:max(obj.q(1:trial)) max(obj.q)+1],trial,run) = sum(S{run}.emissionMatrix(1:C,1:max(obj.q(1:trial))+1,particles,trial),3);
            end
            if obj.transitionMatrixPlot
                P.transitionMatrix(1:C,[1:C novelContext],trial,run) = sum(S{run}.transitionMatrix(1:C,1:C+1,particles,trial),3);
            end

        end
        
        function P = integrate_over_runs(obj,P,S,runWeight)

            if obj.xPredPlot
                P.xPred = obj.weighted_sum_along_dimension(P.xPred,runWeight,4);
            end
            if obj.cPredPlot
                P.cPred = obj.weighted_sum_along_dimension(P.cPred,runWeight,3);
            end
            if obj.xPredPlot && obj.cPredPlot
                P.xHatReduced = obj.weighted_sum_along_dimension(P.xHatReduced,runWeight,2);
            end
            if obj.cFiltPlot
                P.cFilt = obj.weighted_sum_along_dimension(P.cFilt,runWeight,3);
            end
            if obj.cInfPlot
                P.cInf = obj.weighted_sum_along_dimension(P.cInf,runWeight,3);
            end
            if obj.adPlot
                P.retention = obj.weighted_sum_along_dimension(P.retention,runWeight,4);
                P.drift = obj.weighted_sum_along_dimension(P.drift,runWeight,4);
            end
            if obj.bPlot
                P.bias = obj.weighted_sum_along_dimension(P.bias,runWeight,4);
            end
            if obj.emissionMatrixPlot
                P.emissionMatrix = obj.weighted_sum_along_dimension(P.emissionMatrix,runWeight,4);
            end
            if obj.transitionMatrixPlot
                P.transitionMatrix = obj.weighted_sum_along_dimension(P.transitionMatrix,runWeight,4);
            end
            
            if obj.xPredMargPlot
                for run = 1:obj.R
                    P.xPredMarg(:,:,run) = S{run}.xPredMarg;
                end
                P.xPredMarg = obj.weighted_sum_along_dimension(P.xPredMarg,runWeight,3);
            end
            if obj.bPredMargPlot
                for run = 1:obj.R
                    P.bPredMarg(:,:,run) = S{run}.bPredMarg;
                end
                P.bPredMarg = obj.weighted_sum_along_dimension(P.bPredMarg,runWeight,3);
            end
            if obj.yPredMargPlot
                for run = 1:obj.R
                    P.yPredMarg(:,:,run) = S{run}.yPredMarg;
                end
                P.yPredMarg = obj.weighted_sum_along_dimension(P.yPredMarg,runWeight,3);
            end
            if obj.explicitPlot
                for run = 1:obj.R
                    P.explicit(:,run) = S{run}.explicit;
                end
                P.explicit = obj.weighted_sum_along_dimension(P.explicit,runWeight,2);
            end
            if obj.implicitPlot
                for run = 1:obj.R
                    P.implicit(:,run) = S{run}.implicit;
                end
                P.implicit = obj.weighted_sum_along_dimension(P.implicit,runWeight,2);
            end
            if obj.xHatPlot
                for run = 1:obj.R
                    P.xHat(:,run) = S{run}.xHat;
                end
                P.xHat = obj.weighted_sum_along_dimension(P.xHat,runWeight,2);
            end
            for run = 1:obj.R
                P.yHat(:,run) = S{run}.yHat;
            end
            P.yHat = obj.weighted_sum_along_dimension(P.yHat,runWeight,2);

        end
        
        function X = weighted_sum_along_dimension(obj,X,w,dim)

            % find elements that are NaN throughout dimension dim
            i = all(isnan(X),dim);

            % sum over dimension dim of X with weights w
            X = sum(X.*reshape(w,[ones(1,dim-1) numel(w)]),dim,'omitnan');

            % elements that are NaN throughout dimension dim should remain NaN, not 0
            X(i) = NaN;

        end
        
        function P = normalise_relabelled_variables(obj,P,nParticlesUsed,runWeight)

            % normalisation constant
            Z = sum(nParticlesUsed.*runWeight,2);

            if obj.xPredPlot
                P.xPred(:,2:end,:) = P.xPred(:,2:end,:)./Z(1:end-1)';
            end
            if obj.cPredPlot
                P.cPred(2:end,:) = P.cPred(2:end,:)./Z(1:end-1);
            end
            if obj.xPredPlot && obj.cPredPlot
                P.xHatReduced(2:end,:) = P.xHatReduced(2:end,:)./Z(1:end-1);
            end
            if obj.cFiltPlot
                P.cFilt = P.cFilt./Z;
            end
            if obj.cInfPlot
                P.cInf = P.cInf./Z;
            end
            if obj.adPlot
                P.retention = P.retention./Z';
                P.drift = P.drift./Z';
            end
            if obj.bPlot
                P.bias = P.bias./Z';
            end
            if obj.emissionMatrixPlot
                P.emissionMatrix = P.emissionMatrix./reshape(Z,[1,1,obj.T]);
            end
            if obj.transitionMatrixPlot
                P.transitionMatrix = P.transitionMatrix./reshape(Z,[1,1,obj.T]);
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

            RGB = parula(max(P.posteriorMode));
            LW = 2;
            FS = 15;
            
            if obj.xPredPlot
                figure
                YLims = obj.gridX([1 end]);
                YTicks = [0 obj.gridX([1 end])];
                obj.plot_image(P.xPred,YLims,YTicks,[RGB(1:max(P.posteriorMode),:); 0.7 0.7 0.7])
                set(gca,'FontSize',FS)
                ylabel('state | context')
            end
            if obj.cPredPlot
                figure
                hold on
                plot(P.cPred(:,end),'Color',0.7*[1 1 1],'LineWidth',LW)
                for context = 1:max(P.posteriorMode)
                    plot(P.cPred(:,context),'Color',RGB(context,:),'LineWidth',LW)
                end
                axis([0 obj.T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                ylabel('predicted probabilities')
            end
            if obj.xPredPlot && obj.cPredPlot
            end
            if obj.cFiltPlot
                figure
                hold on
                for context = 1:max(P.posteriorMode)
                    plot(P.cFilt(:,context),'Color',RGB(context,:),'LineWidth',LW)
                end
                axis([0 obj.T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                ylabel('responsibilities')
                figure
                plot(P.cFilt(:,end),'Color',0.7*[1 1 1],'LineWidth',LW)
                axis([0 obj.T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                ylabel('novel context probability')
            end
            if obj.adPlot
                figure
                YLims = obj.gridA([1 end]);
                YTicks = [0 obj.gridA([1 end])];
                obj.plot_image(P.retention,YLims,YTicks,RGB)
                set(gca,'FontSize',FS)
                ylabel('retention | context')
                figure
                YLims = obj.gridD([1 end]);
                YTicks = [0 obj.gridD([1 end])];
                obj.plot_image(P.drift,YLims,YTicks,RGB)
                set(gca,'FontSize',FS)
                ylabel('drift | context')
            end
            if obj.bPlot
                figure
                YLims = obj.gridB([1 end]);
                YTicks = [0 obj.gridB([1 end])];
                obj.plot_image(P.bias,YLims,YTicks,RGB)
                set(gca,'FontSize',FS)
                ylabel('bias | context')
                figure;obj.plot_image(P.bias,xLims,nPixels,[-1 1],RGB)
            end
            if obj.transitionMatrixPlot
                for from_context = 1:max(P.posteriorMode)
                    figure
                    hold on
                    plot(squeeze(P.transitionMatrix(from_context,end,:)),'Color',0.7*[1 1 1],'LineWidth',LW)
                    for to_context = 1:max(P.posteriorMode)
                        plot(squeeze(P.transitionMatrix(from_context,to_context,:)),'Color',RGB(to_context,:),'LineWidth',LW)
                    end
                    title(sprintf('\\color[rgb]{%s}context %d',num2str(RGB(from_context,:)),from_context))
                    axis([0 360 -0.1 1.1])
                    set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                    ylabel('transition probabilities')
                end
            end
            if obj.cInfPlot
                figure
                hold on
                plot(P.cInf(:,end),'Color',0.7*[1 1 1],'LineWidth',LW)
                for context = 1:max(P.posteriorMode)
                    plot(P.cInf(:,context),'Color',RGB(context,:),'LineWidth',LW)
                end
                axis([0 obj.T -0.1 1.1])
                set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                ylabel('stationary probabilities')
            end
            if obj.emissionMatrixPlot
                RGBq = cool(max(obj.q));
                for context = 1:max(P.posteriorMode)
                    figure
                    hold on
                    tmp = {};
                    plot(squeeze(P.emissionMatrix(context,end,:)),'Color',0.7*[1 1 1],'LineWidth',LW)
                    tmp = cat(2,tmp,'novel cue');
                    for cue = 1:max(obj.q)
                        plot(squeeze(P.emissionMatrix(context,cue,:)),'Color',RGBq(cue,:),'LineWidth',LW)
                        tmp = cat(2,tmp,sprintf('cue %d',cue));
                    end
                    title(sprintf('\\color[rgb]{%s}context %d',num2str(RGB(context,:)),context))
                    axis([0 360 -0.1 1.1])
                    set(gca,'YTick',[0 0.5 1],'FontSize',FS)
                    ylabel('cue probabilities')
                    legend(tmp,'location','best','box','off')
                end
            end

        end
        
    end
end