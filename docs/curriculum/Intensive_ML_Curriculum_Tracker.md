# Intensive Machine Learning Curriculum Tracker

**Goal:** Build a rigorous foundation in machine learning and neural networks while implementing key algorithms in Rust.

**Study commitment:** 6–8 hours per day, 6 days per week.  
**Core timeline:** 8 weeks.  
**Optional project extension:** 4 additional weeks.

## Daily learning rhythm

- [ ] **2–3 hours:** Course lessons and mathematics.
- [ ] **2–3 hours:** Implement the core idea from scratch in Rust.
- [ ] **1–2 hours:** Exercises, debugging, notes, and review.
- [ ] Keep one day each week for rest or light revision.

## Rules for learning

```text
Understand the equation
  -> work through a tiny numerical example by hand
  -> implement it in Rust
  -> test it on small, inspectable data
  -> use a standard library version
```

## Week 1 — ML Workflow and Linear Regression

**Outcome:** Understand what machine learning is, how models train, and how a line becomes a predictive model.

**Courses/resources:**

- [Kaggle Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Khan Academy Algebra](https://www.khanacademy.org/math/algebra), as needed

- [ ] Complete Kaggle lessons: *How Models Work* and *Basic Data Exploration*.
- [ ] Learn features, labels, predictions, training, validation, and test data.
- [ ] Review functions, lines, slope, and intercept.
- [ ] Understand `prediction = weight * input + bias`.
- [ ] Understand error and mean squared error (MSE).
- [ ] Implement MSE and mean absolute error in Rust.
- [ ] Implement single-input linear regression in Rust.
- [ ] Implement batch gradient descent in Rust.
- [ ] Experiment with several learning rates.
- [ ] Explain why the model updates both weight and bias.

## Week 2 — Vectors, Matrices, and Multiple Features

**Outcome:** Understand the mathematical representation used by larger models and neural-network layers.

**Courses/resources:**

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), linear-algebra material

- [ ] Understand scalars, vectors, and matrices.
- [ ] Calculate vector addition, scalar multiplication, and dot products.
- [ ] Understand matrix dimensions.
- [ ] Implement matrix-vector multiplication in Rust.
- [ ] Understand matrices as linear transformations.
- [ ] Implement multiple linear regression in Rust.
- [ ] Train a model using several features, such as hours studied, sleep, and attendance.
- [ ] Explain `prediction = weights dot features + bias`.
- [ ] Learn and implement feature scaling/normalization.

## Week 3 — Calculus, Derivatives, and Gradients

**Outcome:** Understand why gradient-descent formulas work instead of treating them as magic.

**Courses/resources:**

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
- [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), calculus material

- [ ] Understand derivatives as rates of change.
- [ ] Differentiate simple functions by hand.
- [ ] Understand partial derivatives.
- [ ] Understand the gradient as one partial derivative per parameter.
- [ ] Understand the chain rule conceptually.
- [ ] Derive the weight and bias gradients for linear regression.
- [ ] Explain why gradient descent moves in the opposite direction to the gradient.
- [ ] Add loss-history tracking to the Rust linear-regression project.
- [ ] Test learning rates that converge, learn slowly, and diverge.

## Week 4 — Classification and Probability

**Outcome:** Move from predicting numbers to predicting categories and probabilities.

**Courses/resources:**

- [Machine Learning Specialization — Course 1](https://www.coursera.org/learn/machine-learning)
- [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), probability material

- [ ] Learn logistic regression.
- [ ] Understand the sigmoid function.
- [ ] Understand probabilities and classification thresholds.
- [ ] Understand binary cross-entropy loss.
- [ ] Implement sigmoid in Rust.
- [ ] Implement binary logistic regression with gradient descent in Rust.
- [ ] Learn accuracy, precision, recall, F1 score, and confusion matrices.
- [ ] Test the classifier on a small binary dataset.
- [ ] Explain why MSE is generally not the preferred loss for binary classification.

## Week 5 — Statistics, Validation, and Classical ML

**Outcome:** Learn to judge model quality and choose among simpler ML models.

**Courses/resources:**

- [Kaggle Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

- [ ] Review mean, variance, standard deviation, and normal distributions.
- [ ] Complete Kaggle lessons on validation, underfitting/overfitting, and random forests.
- [ ] Understand bias versus variance.
- [ ] Learn decision trees and random forests.
- [ ] Learn k-nearest neighbors.
- [ ] Learn k-means clustering at a high level.
- [ ] Implement k-nearest neighbors or a small decision tree in Rust.
- [ ] Compare linear regression, a tree model, and k-nearest neighbors on an appropriate dataset.
- [ ] Write a short model-comparison note: accuracy/error, strengths, and weaknesses.

## Week 6 — Neural Networks from First Principles

**Outcome:** Understand a neural network as repeated matrix transformations plus nonlinear activations.

**Courses/resources:**

- [Kaggle Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

- [ ] Understand a neuron: `activation(weights dot inputs + bias)`.
- [ ] Learn ReLU, sigmoid, and softmax.
- [ ] Understand hidden layers and nonlinear decision boundaries.
- [ ] Implement a forward pass through a small two-layer network in Rust.
- [ ] Understand backpropagation conceptually.
- [ ] Work through a backpropagation example with a very small network.
- [ ] Implement a two-layer network in Rust.
- [ ] Train it on XOR or another tiny inspectable dataset.
- [ ] Explain why a single linear model cannot solve XOR.

## Week 7 — Training Neural Networks Well

**Outcome:** Learn the training techniques that make deep models practical.

**Courses/resources:**

- [Kaggle Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
- [fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/), introductory lessons

- [ ] Learn mini-batches, epochs, and stochastic gradient descent.
- [ ] Understand Adam at a high level.
- [ ] Learn dropout and batch normalization.
- [ ] Learn regularization and early stopping.
- [ ] Learn training/validation curves.
- [ ] Add a train/validation split to one project.
- [ ] Identify overfitting from training and validation loss.
- [ ] Train a neural network with a Python framework once, to compare with the Rust version.

## Week 8 — Practical Deep Learning and Capstone Design

**Outcome:** Move from toy data to a real, end-to-end project.

**Courses/resources:**

- [fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)

- [ ] Learn basic tensor operations in PyTorch.
- [ ] Learn datasets, dataloaders, models, loss functions, and optimizers.
- [ ] Choose a real dataset and define a clear prediction task.
- [ ] Create a baseline model.
- [ ] Train at least one improved model.
- [ ] Evaluate it on a held-out test set.
- [ ] Write a short README explaining data, model, metrics, findings, and limitations.
- [ ] Decide whether to use a Rust ML framework, such as Candle or Burn, in a follow-up project.

## Optional Extension — Weeks 9–12

### Week 9 — Capstone implementation

- [ ] Complete data cleaning and exploratory analysis.
- [ ] Implement the chosen baseline.
- [ ] Implement and evaluate an improved model.
- [ ] Track experiments and results.

### Week 10 — Deepen a chosen specialty

Choose one:

- [ ] Computer vision: convolutional neural networks.
- [ ] Natural language processing: embeddings and text classification.
- [ ] Tabular data: boosting, feature engineering, and explainability.
- [ ] Time series: forecasting and sequence validation.

### Week 11 — Rust ML focus

- [ ] Choose a Rust ML library or continue custom implementations.
- [ ] Load a model or weights safely.
- [ ] Run inference from a Rust application.
- [ ] Benchmark or profile one inference workflow.

### Week 12 — Portfolio and review

- [ ] Finish the capstone README and reproducible instructions.
- [ ] Publish or organize the project repository.
- [ ] Write down the topics that still feel unclear.
- [ ] Revisit gradients, backpropagation, probability, and matrices as needed.
- [ ] Choose the next specialization: computer vision, NLP, reinforcement learning, or MLOps.

## Completion milestones

- [ ] I can derive and implement linear-regression gradient descent.
- [ ] I can explain vectors, matrices, dot products, and linear transformations.
- [ ] I can implement logistic regression and evaluate a binary classifier.
- [ ] I can explain and implement a small neural network.
- [ ] I can train and evaluate a practical model with PyTorch.
- [ ] I have one finished end-to-end project.

## Topics to postpone

You do not need these to finish this curriculum:

- Differential equations
- Partial differential equations (PDEs)
- Abstract algebra
- Rigorous real analysis
- Advanced optimization theory
