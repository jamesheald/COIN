SR = COIN;

SR.perturbations = [zeros(1,50) ones(1,125) -ones(1,15) NaN(1,150)];
SR.runs = 10;
SR.plot_state_given_context = true;
SR.plot_predicted_probabilities = true;
SR.plot_state = true;

fprintf('running the COIN model on the spontaneous recovery paradigm, number of runs = %d\n',SR.runs')

S = SR.simulate_COIN;
P = SR.plot_COIN(S);