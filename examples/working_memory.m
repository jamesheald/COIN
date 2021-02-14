WM = COIN;

WM.perturbations = [zeros(1,192) ones(1,384) -ones(1,20) NaN(1,192)];
WM.eraser_trials = 597;
WM.runs = 10;
WM.plot_state_given_context = true;
WM.plot_predicted_probabilities = true;
WM.plot_state = true;

fprintf('running the COIN model on the spontaneous recovery paradigm with working memory task, number of runs = %d\n',WM.runs')

S_WM = WM.simulate_COIN;
P_WM = WM.plot_COIN(S_WM);

fprintf('running the COIN model on the spontaneous recovery paradigm without working memory task, number of runs = %d\n',WM.runs')

WM.eraser_trials = [];

S = WM.simulate_COIN;
P = WM.plot_COIN(S);