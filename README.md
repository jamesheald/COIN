# COIN

The COIN (COntextual INference) model is a principled Bayesian model of motor learning in which separate memories are stored for different contexts. Each memory stores information learned about the dynamical and sensory properties of the environment associated with the corresponding context. The key insight is that memory creation, updating, and expression are all controlled by a single computation&mdash;contextual inference.

The model can account for key features of motor learning that had no unified explanation: spontaneous recovery, savings, anterograde interference, how environmental consistency affects learning rates and the distinction between explicit and implicit learning.

The COIN model can be used to model trial-based learning in force-field and visuomotor adaptation paradigms. 

The model is available for MATLAB and has been tested on Matlab R2020a.



BADS is a novel, fast Bayesian optimization algorithm designed to solve difficult optimization problems, in particular related to fitting computational models (e.g., via maximum likelihood estimation).

BADS has been intensively tested for fitting behavioral, cognitive, and neural models, and is currently being used in many computational labs around the world. In our benchmark with real model-fitting problems, BADS performed on par or better than many other common and state-of-the-art MATLAB optimizers, such as fminsearch, fmincon, and cmaes [1].

BADS is recommended when no gradient information is available, and the objective function is non-analytical or noisy, for example evaluated through numerical approximation or via simulation.



A description of your project follows. A good description is clear, short, and to the point. Describe the importance of your project, and what it does.


### Reference

1. Heald J, Lengyel M, Wolpert D. 2020. Contextual inference underlies the learning of sensorimotor repertoires. *bioRxiv* doi: 10.1101/2020.11.23.394320 ([link to preprint](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1))

## Installation

1. Download the COIN.m file.
2. Install the following packages (to improve the efficiency of the code):
   - "[Lightspeed matlab toolbox](https://github.com/tminka/lightspeed)" by Tom Minka. 
     - Run install_lightspeed.m to compile the necessary mex functions. 
   - "[Nonparametric Bayesian Mixture Models - release 2.1](http://www.stats.ox.ac.uk/~teh/software.html)" by Yee Whye Teh.
     - Run make.m to compile the necessary mex functions. 
   - "[Truncated Normal Generator](https://web.maths.unsw.edu.au/~zdravkobotev/)" by Zdravko Botev.

Add each package to the MATLAB search path using the following command: 
```
addpath(genpath('directory')); % 'directory' is the full path to the root folder of the package
```

## Usage

Instructions on how to use the COIN model are provided on the [wiki](https://github.com/jamesheald/COIN/wiki).

## Contact information

Feel free to e-mail me at [jamesbheald@gmail.com](mailto:jamesbheald@gmail.com) if you have any questions or encounter a problem with the code.

## License

COIN is released under the terms of the GNU General Public License v3.0.
