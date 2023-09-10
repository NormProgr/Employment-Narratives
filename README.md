# Employment Narratives

This is the most recent branch containing the content to be graded.

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

## Research proposal

In my research proposal I want to give a first idea how feasible it is to use news
outlets data and NLP methods to evaluate the publics perception about labor topics.
Applying these methods allows for new insights about possible information asymmetries
that influence individuals decision making and bargaining about wages, employment and
labor market frictions. To conduct this, I classify the news outlets text data into
three (non-exclusive) categories: Labor Supply, Labor Demand, and Government
Intervention. These Categories should show which reasons the news outlets identify for
Labor market related issues. In this context, I draw inspiration from Andre et al.'s
(2023) paper on "Inflation Narratives," which manually categorizes narratives related to
inflation. My primary contribution is automating the classification of considerably
larger datasets. This endeavor aligns with the idea of harnessing extensive language
data for economic analysis, as advocated by Sendhil Mullainathan and Jann Spiess (2017).

The influence of employment narratives on individuals' decision-making and bargaining
has already been explored in studies by Kennan (2010) and Walter (2017). Kennan (2010)
identifies private information as a valid explanation for fluctuations in unemployment.
These findings have varying implications, depending on whether employers or employees
possess better information. Walter (2017) delves into how globalization impacts
individuals' perceptions of labor market frictions and resultant policy shifts. Both
studies emphasize the significance of individuals' information in shaping real economic
indicators. My work aligns closely with Garz's 2012 paper, which examines the media's
coverage and its impact on job insecurity in labor market policies. Garz (2012) analyzes
handpicked news articles from German media and reveals that a poorly performing labor
market and high news volume contribute to heightened insecurity perceptions,
underscoring the importance of understanding microeconomic expectation formation.

My research proposal comprises three main sections: I. Providing a concise overview of
my initial investigation. II. Addressing challenges encountered, discussing the dataset,
and assessing project feasibility. III. Presenting the results and offering additional
insights into the methods and remaining issues.

I. Overview

I have gathered scraped data from CNN archives from
[Kaggle](https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning).
The data is spanning from 2011 to 2022, totaling 42,000 unfiltered and unsorted articles
encompassing topics such as world news, sports, business, economics, and health. For
analysis, I employ the
[Valhalla/distilbart-mnli-12-6](https://huggingface.co/valhalla/distilbart-mnli-12-6)
transformer model for zero-shot classification, processing a curated subset of 24,000
data points after filtering out categories like sports, entertainment, and NAs. Given
the sheer volume of news article text data, I opt to work with shorter summaries
(description text) of the articles to ensure reasonable processing times. While the
Facebook/bart-large-mnli model was initially considered and widely used in the Hugging
Face community for zero-shot classification, I ultimately select the
[Valhalla/distilbart-mnli-12-6](https://huggingface.co/valhalla/distilbart-mnli-12-6)
model due to its efficient performance, maintaining 90% accuracy while significantly
reducing computation time.

To gauge the effectiveness of zero-shot classification, I manually classify a random
subset of data points and compare the results with those generated by the zero-shot
classification method. The zero-shot classified data subsequently serves as the training
dataset for a [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
model, a more compact and faster version of BERT, well-suited for fine-tuning on
specific tasks, aligning with my research goals. The trained model can then be applied
to new, unseen data for similar tasks, although I won't delve into that aspect in this
proposal.

II Findings and Discussion

Upon applying zero-shot classification, I attain an average accuracy of 77.78%, with
substantial variations across the different classes. However, these results present
certain challenges. Notably, government intervention is the most frequently classified
category (10 out of 100), while labor supply is never identified, achieving a perfect
accuracy score of 100%. These findings underscore the presence of an imbalanced data
issue. I attribute this imbalance to two primary factors: Firstly, government
intervention is a more frequently encountered semantic category compared to labor supply
and demand. Secondly, CNN encompasses a wide spectrum of news categories, including many
related to sports and entertainment. To mitigate the impact of this issue, I implement
data cleaning procedures, ensuring both a substantial dataset in terms of quantity and
per text. I retain the term "government intervention" due to the potential influence of
various government interventions on labor market perceptions, although the actual impact
on labor markets remains subject to debate. Nevertheless, despite these measures, the
data imbalance persists as a significant challenge.

My second model is based on a pretrained BERT architecture, which I fine-tuned using a
zero-shot labeled dataset. During this process, I encountered some challenges related to
the non-exclusive class design I initially adopted. Consequently, I had to modify the
forward method to better suit this setup. The primary distinction lies in how the loss
function is calculated and, consequently, how the final layer logits are generated. In
exclusive class classification, one typically employs a Softmax Cross-Entropy Loss
function to compute a probability distribution for each class, ensuring that the
probabilities sum up to 1. However, my approach handles multilabel classification, where
an input can belong to multiple classes simultaneously. Therefore, I implemented the
Binary Cross-Entropy Loss function (BCE). BCE treats each label within the multilabel
class as a binary variable, allowing for values greater than 1 for the entire class.
Finally, I receive my training results running a subset of 1000 data points an accuracy
of 82,16% at the third epoch. Considering that the underlying training data might be
imbalanced I consider the model to perform quite well on the data at hand. Though both
models are distilbert based, which might account for the high precision. But including
my zero-shot classification evaluation results I think the overall data is not suitable
to solve my problem or being applied on new unseen data. In the next part I present some
ideas how to overcome this problem.

III Outlook and further Ideas

- The need to train LLMs on economic related news (e.g. the Economist) and articles
  (e.g. American Economic Review Journals) to get a better understanding of economic
  principles semantics :bulb:
  - This approach might allow enhance my project because more precise labeling classes
    for labor related narratives are possible, like Andre et al. (2023) use for
    Inflation narratives. :memo:
- I could consider other sources of information besides of classic news media like X
  (Twitter) or LinkedIn Data :memo:
- For Inference: link results of labor market papers with the my research approach to
  get an idea how correct are news understanding labor market topics. :memo:

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
      able to run the Code on a GPU.`import torch` and
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
      optimization iteration. It may also lead to noisier gradients as the small batches
      do not represent the data well.
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
      semantic meaning, while the decoder translates. Unlike simpler encoders they can
      handle input and output variables with differing length.
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

|     | Class Accuracy                   | Mean Accuracy | Class Name                                                    | Count Ones   |
| --: | :------------------------------- | ------------: | :------------------------------------------------------------ | :----------- |
|   0 | \[0.4, 0.9333333333333333, 1.0\] |      0.777778 | \['government intervention', 'labor demand', 'labor supply'\] | \[10, 1, 0\] |

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

## References

- Andre, Peter and Haaland, Ingar and Roth, Christopher and Wohlfart, Johannes,
  Narratives About the Macroeconomy (2023). CESifo Working Paper No. 10535, Available at
  SSRN: https://ssrn.com/abstract=4506321 or http://dx.doi.org/10.2139/ssrn.4506321
- Garz, M. Job Insecurity Perceptions and Media Coverage of Labor Market Policy. J Labor
  Res 33, 528–544 (2012). https://doi.org/10.1007/s12122-012-9146-9
- Kennan, John, Private Information, Wage Bargaining and Employment Fluctuations, The
  Review of Economic Studies, Volume 77, Issue 2, April 2010, Pages 633–664,
  https://doi.org/10.1111/j.1467-937X.2009.00580.x
- Mullainathan, Sendhil, and Jann Spiess. "Machine learning: an applied econometric
  approach." Journal of Economic Perspectives 31.2 (2017): 87-106.
- Walter, S. (2017). Globalization and the Demand-Side of Politics: How Globalization
  Shapes Labor Market Risk Perceptions and Policy Preferences. Political Science
  Research and Methods, 5(1), 55-80. doi:10.1017/psrm.2015.64
