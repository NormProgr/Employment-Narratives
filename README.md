# Employment Narratives

## Scope of Project

I provide an economic application to Employment Narratives in global news outlets like
CNN (Cable News Network) or New York Times and social networks like Twitter. For that I
will scrape data from Twitter and use some data from
[Kaggle Datasets](https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning)
to access CNN and/or New York Times. After the data is cleaned I will use an NLP model
for that to work I will classify the data into self set classes like labor supply, labor
demand, and labor market government interventions. Afterwards I will benchmark the
performance by testing it on a hand-labelled set. The goal is to examine what reasons
news outlets see for labor market business cycles or criseses at the time they are
happening. An interesting extension would be how gould these news are correlating with
economic papers explaining these phenomenons. This could indicate whether news have
certain biases explaining labor market phenomenons and how well they perform to evaluate
them.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/R1vgPUT1)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/NormProgr/EN/main.svg)](https://results.pre-commit.ci/latest/github/NormProgr/EN/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create and activate the environment with

```console
$ git clone https://github.com/iame-uni-bonn/final-project-NormProgr.git
$ cd final-project-NormProgr
$ conda env create -f environment.yml
$ conda activate EN
$ pre-commit install
```

To build the project, type

```console
$ pytask
```

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
