# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning a motor repertoire in which separate memories are stored for different contexts. The model is described in detail in a recent bioRxiv preprint: [Contextual inference underlies the learning of sensorimotor repertoires](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1).

## Installation

1. Download the COIN.m file.
2. Install the following packages (to improve the efficiency of the code):
   - "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. 
     - Run install_lightspeed.m to compile the necessary mex functions. 
   - "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh.
     - Run make.m to compile the necessary mex functions. 
   - "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.
3. Install:
   - "[Bayesian Adaptive Direct Search (BADS)](https://github.com/lacerbi/bads)" by Luigi Acerbi. 
Add each package to the MATLAB search path using the following command: 
```
addpath(genpath('directory'));
```
## Installation

## License

The COIN model is released under the terms of the GNU General Public License v3.0.
