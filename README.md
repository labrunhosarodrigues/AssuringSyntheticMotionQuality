# motiongen

[Original Paper](https://www.scitepress.org/Papers/2026/144405/144405.pdf)

This project contains the code used in the development of the paper "Controlled Large Scale Synthetic Motion Dataset Generation Leveraging Text-to-Motion and Sample-wise Quality Assurance".

It contains scripts for 3 different parts:
 - Generating MDM motion samples.
 - Getting MoBERT scores from sets of generated samples.
 - Performing the binary search process described in the paper.
 - Training and evaluating an activity recoginition model with the generated data.

## Instalation

This project requires 3 different environments to run:
     - **mdm**: created following MDM's setup instructions, its needed to use MDM's generative model.
     - **motiongen**: created using motiongen_env.yml (conda), its used to run the notebooks, MoBERT, and sample quality verification program.
     After creating this environment, activate it and run `pip install git+https://github.com/labrunhosarodrigues/MoBERT.git@make-pip-installable` to install MoBERT. To finish its setup, you must download the model [checkpoint](https://drive.usercontent.google.com/download?id=1gmljNRJKf_IujUIlcmCl9Q6mZI_Qceiv&export=download&authuser=0) and run `python -c "mobert.primary_evaluator.save_primary_evaluator_archive(<path_to_downloaded_file>)"`.
     - **motion_training**: created using motion_training_env.yml (conda), its used to run the training of the motion recognition models

## Usage

To generate the pilot and large dataset with MDM, activate the mdm environment and use the commands available under `python -m mdm_aux`.

To run the notebooks under `analysis` folder, use the environment motiongen for the kernel.

To use MoBERT to evaluate a set of generated samples, use the command `python -m motiongen.mobert_eval`.

To perform the quality assurance procedure, after collecting MoBERT scores, run `python -m motiongen.quality_checking`.

Finally, to perform training on the filtered dataset, activate the motion_training environment and run `motiongen.activity_recognition.motion_training`.

## Citation

If you cite this work, please use the following:

```bib
@conference{biosignals26,
author={Lourenço Rodrigues and Markus Wenzel and Felix Putze},
title={Controlled Large Scale Synthetic Motion Dataset Generation Leveraging Text-to-Motion and Sample-Wise Quality Assurance},
booktitle={Proceedings of the 19th International Joint Conference on Biomedical Engineering Systems and Technologies - Volume 1: BIOSIGNALS},
year={2026},
pages={140-150},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0014440500004070},
isbn={978-989-758-802-0},
}
```

## Disclamer

Copyright (C) 2024 Cognitive Systems Lab.