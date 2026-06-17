# AI-Rossby Project outline

The objective of this repository is to create a unified codebase for training, validating, profiling, and predicting with AI weather and climate emulators in pytorch. The objectives of this codebase are for it to be:

- As modular as possible so that new models, training methods, inference methods, prediction metrics can be easily created from the base classes
- Contain a robust set of unit and smoke tests that users can use to validate any new modules or classes
- Contain Claude skills that users can give to their own claude instances to ease development, testing, and model optimization
- Use a robust system of configuration files that contain all information relevant to different models, data sets, training parameters, and validation parameters in separate and distinct sections/files
- As easy to read as possible for scientists new to the field
- Use optimized pipelines and kernels wherever possible, drawing from known optimized repositories (i.e., makani)
- Be easily implementable on different compute systems that use job schedulers such as PBS and SLURM

## Current Repositories

This repository should draw from the following existing repositories:
- PanguWeather/v2.0 (located at /Users/Alexander/Documents/UChicago/PanguWeather) - this is the current working repository for the AI emulators built and run by our research group. It is messy and not well modularized, but it will be necessary to design helper scripts to translate between the current data configurations, model state definitions, and model checkpoints and their final versions in this repository.
- amip (located at /Users/Alexander/Documents/UChicago/amip) - this is our collaborators repository for training and validating their stochastic interpolant-based latent diffusion models for climate emulations, written in torch-lightning. Helper scripts are again needed to translate between the expected data set, models and checkpoints from this repo and the ai-rossby. Additionally, all modular classes designed in this repo must be able to represent the stochastic interpolant-based emulators.
- physics-nemo (located at /Users/Alexander/Documents/UChicago/physicsnemo) - this is Nvidia's state-of-the-art training library for physics-informed AI models, and contains capabilities for training many of the state of the art weather models as well as diffusion-type models.


### First objective
I would like to port all of the current models, training and inference code, and mid-training inference validation and after the fact inference validation in PanguWeather/v2.0 and amip into the frameworks of physics-nemo.

You need to be the expert in doing this. Start by thoroughly analyzing these repositories and coming up with an implementation plan for the physicsnemo implementation. As much as possible, build off of what already exists. If anything requires my input, come back with a set of questions on how to proceed.
