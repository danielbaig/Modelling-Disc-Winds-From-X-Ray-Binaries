# Modelling Disc Winds From X-Ray Binaries
This project was undertaken in the summer of 2023. Its aim was to reproduce P-Cygni profiles in the optical spectrum in an X-ray binary as seen in [MAXI J1820](https://iopscience.iop.org/article/10.3847/2041-8213/ab2768). The simulations were run using the Monte Carlo radiative transfer code: [PYTHON](https://github.com/agnwinds/python). 

This project mainly consists of data analysis and data visulisation. There is an additional section on string minimisation, which is a possible method to emulate points in the parameter space without the need to simulate them.

## Operation
Everything (with the exception of the C++ code) can be run from the main.py file in the terminal at the head of the project. This includes: analysis of a single run, comparing two runs, comparing all runs, prepareing data for the string minimisation procedure and analysing data from the string minimisation procedure.

If PYTHON runs are to be added, their files should be placed in the ```/data/``` directory named in the form `XRBi' where `i' is an index. Note: in the code's current state some functionality relys on indicies starting at 0 and not having any gaps, i.e., 0,1,2,...20.

### Features
* Has the ability to create detailed pairwise correlation plots of the highly dimensional parameter spaces.
* Uses an implementation of the Barnes Hut t-SNE to visulise a projection of the highly dimensional parameter spaces.
* Can create rotated-renormalised quantile-quantile comparisons of data ponts.
* Can create zoomed in spectra.
* Can produce a flowfield plot overlayed on a electron temperature heatmap of the disc wind.
* Can perform the string minimisation procedure.
