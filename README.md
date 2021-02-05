# COIN

The COIN (COntextual INference) model is a principled Bayesian model of learning in which separate memories are acquired for different contexts. It has been used in the motor domain to model learning in force-field and visuomotor adaptation paradigms. 

The COIN model was developed using MATLAB and has been tested on Matlab R2020a.

### Reference

1. Heald J, Lengyel M, Wolpert D. 2020. Contextual inference underlies the learning of sensorimotor repertoires. *bioRxiv* doi: 10.1101/2020.11.23.394320 ([preprint](https://www.biorxiv.org/content/10.1101/2020.11.23.394320v1))

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
