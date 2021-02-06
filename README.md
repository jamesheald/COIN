# COIN

The COIN (COntextual INference) model [[1](#reference)] is a principled Bayesian model of motor learning in which separate memories are acquired for different contexts. The key insight of the model is that memory creation, updating, and expression are all controlled by a single computation&mdash;contextual inference.

The model was developed in MATLAB and has been tested on Matlab R2020a.

&nbsp;
<p align="center">
<img src="https://github.com/jamesheald/COIN/blob/main/spotaneous_recovery.png" width="633.5000" height="361.0000">
</p>
&nbsp;

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
3. Add each package to the MATLAB search path using the following command: 
    ```
    addpath(genpath('directory')); % 'directory' is the full path to the base folder of the package
    ```
    
## Usage

Instructions for how to use the COIN model can be found on the [COIN wiki page](https://github.com/jamesheald/COIN/wiki).

## Contact information

Feel free to e-mail me at [jamesbheald@gmail.com](mailto:jamesbheald@gmail.com) if you have any questions or encounter any problems with the code.

## License

The COIN model is released under the terms of the GNU General Public License v3.0.
