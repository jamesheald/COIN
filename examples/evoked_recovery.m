ER = COIN;

ER.perturbations = [zeros(1,50) ones(1,125) -ones(1,15) NaN NaN 1 1 NaN(1,148)];
ER.runs = 10;
ER.plot_state_given_context = true;
ER.plot_predicted_probabilities = true;
ER.plot_state = true;

fprintf('running the COIN model on the evoked recovery paradigm, number of runs = %d\n',ER.runs')

S = ER.simulate_COIN;
P = ER.plot_COIN(S);