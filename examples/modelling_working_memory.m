% paradigm based on Keisler, A. & Shadmehr, R. A shared resource between declarative memory and motor memory. J. Neurosci. 30, 14817–14823 (2010).

obj_WM = COIN;
obj_WM.perturbations = [zeros(1,192) ones(1,384) -ones(1,20) NaN(1,192)];
obj_WM.runs = 10;
obj_WM.max_cores = feature('numcores');
obj_WM.plot_state_given_context = true;
obj_WM.plot_predicted_probabilities = true;
obj_WM.plot_state = true;

% for a working memory task performed between trials 596 and 597, set the
% predicted probabilities on trial 597 to the stationary context 
% probabilities
obj_WM.stationary_trials = 597;

OUTPUT_WM = obj_WM.simulate_COIN;
