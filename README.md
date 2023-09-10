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

[![pre-commit.ci passed](https://img.shields.io/badge/pre--commit.ci-passed-brightgreen)](https://results.pre-commit.ci/run/github/274689747/1678058970.SI-lnarDSRqXafVBdLucmg)
[![image](https://img.shields.io/badge/pytask-v0.3.1-red)](https://pypi.org/project/pytask/)
[![image](https://img.shields.io/badge/python-3.11.0-blue)](https://www.python.org/)
[![image](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/license/mit/)

## Usage

To get started, create and activate the environment with

```console
$ git clone https://github.com/iame-uni-bonn/final-project-NormProgr.git
$ cd ./final-project-NormProgr
$ conda env create -f environment.yml
$ conda activate EN
$ pre-commit install
```

In addition, you need to have a kaggle account and an API token. You can find
Instruction on how to get one [here](https://www.kaggle.com/docs/api). Then, you need to
download your personalized token called `kaggle.json` in the `users/<username>/.kaggle/`
folder.

To build the project, type

```console
$ pytask
```

## Configuration

The code makes an automated check whether or not you use a CPU or GPU. With a CPU
automatically a smaller dataset of 100 will be executed. Further configurations are
stored in `src/EN/analysis/model_config.yaml`. You can freely change the initial
configurations by:

```console
classifiable_data: Description # can be changed to "Article text" or keep "Description"
model_name: valhalla/distilbart-mnli-12-6
batch_size: 20
epochs: 3
weight_decay: 0.01
model_name_pred: distilbert-base-uncased
```

Additionally it is possible to change the `seed = 42` which is not recommended if you
did not classify a subset of data by hand by yourself for the zero-shot evaluation.

## Questions & Answers

1. List five different tasks that belong to the field of natural language processing.
   1. Text classification
   1. Named Entity Recognition
   1. Machine Translation
   1. Text Generation
   1. Question Answering
1. What is the fundamental difference between econometrics/statistics and suprevised
   machine learning
   1. Econometrics estimates unobservable parameters and tests hypotheses about them.
      Therefore, one does not know how well it worked. Whereas supervised machine
      learning predicts observable things and can check/evaluate how well it worked.
1. Can you use stochastic gradient descent to tune the hyperparameters of a random
   forrest. If not, why?
   1. No, SGD can be just applied to functions that are differentiable. Random forests
      are not differentiable because they are based on decision trees which are binary
      and therefore not differentiable. Additionally, both concepts address different
      problems. SGD optimizes the parameters of a single model through iterative
      updates, while Random Forrest hyperparameters determine ensemble behavior of many
      decision trees e.g. through grid search.
1. What is imbalanced data and why can it be a problem in machine learning?
   1. Imbalanced data describes a dataset with unequal distribution of
      classes/categories. It can lead to overfitting, loss of important info or a bias
      towards the majority class.
1. Why are samples split into training and test data in machine learning?
   1. Samples are split into training and test data to assess how well a trained model
      generalizes to new, unseen data. This way the applied model can be evaluated.
1. Describe the pros and cons of word and character level tokenization.
   1. Word tokenization is simple, preserves the word structure and there are not too
      many words to deal with. But it has problems to deal with typos, word variation,
      morphologies or incomplete words. Whereas Character tokenization is even simpler,
      has a tiny vocabulary size and there are no unknown words but the tokenized texts
      are longer and it loses the entire word structure.
1. Why does fine-tuning usually give you a better performing model than feature
   extraction?
   1. This is because in fine-tuning, all model parameters, including the last hidden
      state, are optimized for the task, whereas in feature extraction, only the
      classifier is trained, the performance is then depending on the potentially
      coincidental relevance of the last hidden state.
1. What are advantages over feature extraction over fine-tuning.
   1. Feature extraction is less computationally intensive and can run on a CPU. On the
      other hand, fine-tuning, due to its slowness on a CPU, requires a GPU for
      efficient processing.
1. Why are neural networks trained on GPUs or other specialized hardware?
   1. Neural Networks are inherent parallel algorithms. GPUs have much more cores (even
      though smaller ones) than CPUs and are better at handling a lot of parallelized
      tasks.
1. How can you write pytorch code that uses a GPU if it is available but also runs on a
   laptop that does not have a GPU.
   1. If you run the following code and check which state is true GPU or CPU. You are
      able to run it on all machines.`import torch` and
      `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
1. How many trainable parameters would the neural network in this video have if we
   remove the second hidden layer but leave it otherwise unchanged.
   1. There should be 12730 trainable parameters if the second hidden layer is excluded.
1. Why are nonlinearities used in neural networks? Name at least three different
   nonlinearities.
   1. Without nonlinearities we would simply have a linear model that would not be able
      to capture the complex relationships in a lot of data.
   1. Three Nonlinearities: Rectified Linear Unit (ReLU), Sigmoid, Hyperbolic Tangent
      (tanh)
1. Some would say that softmax is a bad name. What would be a better name and why?
   1. "SoftArgmax" would be a more appropriate term because it indicates the underlying
      mathematical operation. The "Softmax" function isn't a softened version of the
      maximum itself, but rather a softened version of the one-hot encoded "argmax."
1. What is the purpose of DataLoaders in pytorch?
   1. DataLoaders makes it easy to loop over batches of the data and the batches can be
      loaded in parallel.
1. Name a few different optimizers that are used to train deep neural networks
   1. SGD (stochastic gradient descent)
   1. SDG + Momentum
   1. Adam (Adaptive Moment Estimation)
   1. RMSprop (Root Mean Square Propagation)
   1. Adagrad (Adaptive Gradient Algorithm)
1. What happens when the batch size during the optimization is set too small?
   1. When the batch size becomes too small, updates to the model's parameters become
      erratic, leading to unpredictable and inconsistent changes during each
      optimization iteration. But also the optimization is computational fast.
1. What happens when the batch size diring the optimization is set too large?
   1. Too large batch sizes lead to overfitting of the model. It also requires a lot
      more computational power.
1. Why can the feed-forward neural network we implemented for image classification not
   be used for language modelling?
   1. For language modelling we need neural networks that are able to capture sequential
      data by maintaining hidden states that capture context over time. In a
      feed-forward neural network information just flows in one direction, which is
      enough for image classification that has no context over time (at least in our
      case).
1. Why is an encoder-decoder architecture used for machine translation (instead of the
   simpler encoder only architecture we used for language modelling)
   1. The encoder-decoder architecture is employed in machine translation to
      comprehensively understand source sentences bidirectionally. The encoder captures
      semantic meaning, while the decoder translates. Unlike simpler encoders, this
      approach grasps context, idiomatic expressions, and linguistic nuances.
1. Is it a good idea to base your final project on a paper or blogpost from 2015? Why or
   why not?
   1. If the paper or blogpost had computational or methodological limitations but
      interesting data it would be interesting to solve these issues now (e.g. handling
      machine translation). But using methods from 2015 would limit the analysis of my
      final paper as the scientific improvements were significant the last years
      (especially because of transformers).
1. Do you agree with the following sentence: To get the best model performance, you
   should train a model from scratch in Pytorch so you can influence every step of the
   process.
   1. Disagree. Pre-built models are often numerically optimized and computationally
      efficient. Creating and pre-training a new model from scratch is time-consuming.
      However, customizing specific components can be valuable. Combining pre-trained
      models with tailored elements strikes a balance between efficiency and
      customization in model development.
1. What is an example of an encoder-only model?
   1. BERT is an example of an encoder-only model.
1. What is the vanishing gradient problem and how does it affect training?
   1. The vanishing gradient problem occurs in neural networks (like RNN) when gradients
      of the loss function become extremely small during training, hindering updates to
      early layers. It affects training by preventing these layers from learning complex
      patterns, leading to suboptimal performance.
1. Which model has a longer memory: RNN or Transformer?
   1. Transformers generally have longer memory compared to RNN. This is due to the
      vanishing gradient problems of RNN.
1. What is the fundamental component of the transformer architecture?
   1. The fundamental component is the “Attention” mechanism.

## Appendix

### Appendix A

Zeo-Shot Evaluation:

|     | Class Accuracy                                   | Mean Accuracy | Class Name                                                    | Count Ones   |
| --: | :----------------------------------------------- | ------------: | :------------------------------------------------------------ | :----------- |
|   0 | \[0.35714285714285715, 0.9285714285714286, 1.0\] |      0.761905 | \['government intervention', 'labor demand', 'labor supply'\] | \[10, 1, 0\] |

### Appendix B

Training size 100:

|     | trained                                                                                                                                                                                                               | eval.eval_loss | eval.eval_accuracy_thresh | eval.eval_runtime | eval.eval_samples_per_second | eval.eval_steps_per_second | eval.epoch |
| --: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------: | ------------------------: | ----------------: | ---------------------------: | -------------------------: | ---------: |
|   0 | TrainOutput(global_step=12, training_loss=0.38649340470631915, metrics={'train_runtime': 47.622, 'train_samples_per_second': 5.04, 'train_steps_per_second': 0.252, 'train_loss': 0.38649340470631915, 'epoch': 3.0}) |       0.317368 |                  0.816667 |            0.9154 |                       21.849 |                      1.092 |          3 |

### Appendix C

Training size 1000:

|     | trained                                                                                                                                                                                                                | eval.eval_loss | eval.eval_accuracy_thresh | eval.eval_runtime | eval.eval_samples_per_second | eval.eval_steps_per_second | eval.epoch |
| --: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------: | ------------------------: | ----------------: | ---------------------------: | -------------------------: | ---------: |
|   0 | TrainOutput(global_step=120, training_loss=0.3159986178080241, metrics={'train_runtime': 912.944, 'train_samples_per_second': 2.629, 'train_steps_per_second': 0.131, 'train_loss': 0.3159986178080241, 'epoch': 3.0}) |       0.304869 |                  0.821667 |           13.9332 |                       14.354 |                      0.718 |          3 |

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
