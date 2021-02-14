SR = COIN;

SR.perturbations = [zeros(1,100) ones(1,200) -ones(1,20) NaN(1,100)];
SR.runs = 10;
SR.infer_bias = true;
SR.plot_state_given_context = true;
SR.plot_bias_given_context = true;
SR.plot_predicted_probabilities = true;
SR.plot_state = true;
SR.plot_bias = true;
SR.plot_state_feedback = true;
SR.plot_explicit_component = true;
SR.plot_implicit_component = true;

fprintf('running the COIN model on the spontaneous recovery paradigm with explicit reporting, number of runs = %d\n',SR.runs')

S = SR.simulate_COIN;
P = SR.plot_COIN(S);

line_width = 2;
font_size = 15;

figure
hold on
plot(P.explicit_component,'Color',[0.2000 0.6275 0.1725],'LineWidth',line_width)
plot(P.implicit_component,'Color',[0.6980 0.8745 0.5412],'LineWidth',line_width)
plot(P.implicit_component+P.explicit_component,'Color',[0.9843 0.6039 0.6000],'LineWidth',line_width)
plot(P.average_state_feedback,':','Color',[0.9843 0.6039 0.6000],'LineWidth',line_width)
set(gca,'YTick',[-1 0 1],'FontSize',font_size)
axis([0 420 -1.1 1.1])
legend('explicit','implicit','total','total (control)','location','best')
legend box off
box off