% paradigm adapted from McDougle, S. D., Bond, K. M. & Taylor, J. A. Explicit and implicit processes constitute the fast and slow processes of sensorimotor learning. J. Neurosci. 35, 9568–9579 (2015).

obj_SR = COIN;
obj_SR.perturbations = [zeros(1,100) ones(1,200) -ones(1,20) NaN(1,100)];
obj_SR.runs = 10;
obj_SR.max_cores = feature('numcores');

% infer the measurment bias for a visuomotor rotation experiment
obj_SR.infer_bias = true;

obj_SR.plot_state_given_context = true;
obj_SR.plot_bias_given_context = true;
obj_SR.plot_predicted_probabilities = true;
obj_SR.plot_state = true;
obj_SR.plot_bias = true;
obj_SR.plot_state_feedback = true;
obj_SR.plot_explicit_component = true;
obj_SR.plot_implicit_component = true;

OUTPUT = obj_SR.simulate_COIN; 

figure
hold on
line_width = 2;
font_size = 15;
plot(OUTPUT.plots.explicit_component(1:320),'Color',[0.2000 0.6275 0.1725],'LineWidth',line_width)
plot(OUTPUT.plots.implicit_component,'Color',[0.6980 0.8745 0.5412],'LineWidth',line_width)
plot(OUTPUT.plots.implicit_component(1:320)+OUTPUT.plots.explicit_component(1:320),'Color',[0.9843 0.6039 0.6000],'LineWidth',line_width)
plot(OUTPUT.plots.average_state_feedback,':','Color',[0.9843 0.6039 0.6000],'LineWidth',line_width)
set(gca,'XTick',[0 420],'YTick',[-1 0 1],'FontSize',font_size)
axis([0 420 -1.1 1.1])
legend('explicit component','implicit component','total adaptation when explicit report solicited','total adaptation when no explicit report solicited (control)','location','best')
legend box off
ylabel('adaptation')
xlabel('trial')
box off
