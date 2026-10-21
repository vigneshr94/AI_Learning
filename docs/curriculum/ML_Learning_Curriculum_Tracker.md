# Machine Learning Learning Tracker

**Goal:** Understand machine-learning foundations deeply enough to implement core algorithms in Rust, then apply them using standard ML tools.

**Suggested pace:** 6–8 hours per week.  
**Target duration:** 20 weeks, followed by project-based study.

## How to use this tracker

- Mark a task complete with `[x]` when you can explain it and have completed the related exercise.
- Use Python-based courses for their explanations and ecosystem, then reimplement the central algorithm in Rust.
- Do not wait to "finish all math" before building models. Learn each mathematical concept when it becomes useful.

## Learning sequence

```text
Algebra and functions
  -> Vectors and matrices
  -> Derivatives and gradients
  -> Linear and logistic regression
  -> Neural networks and backpropagation
  -> Modern deep learning
```

## Phase 1 — Machine Learning Orientation (Weeks 1–2)

**Outcome:** Know the standard ML workflow and the vocabulary used in every later phase.

**Primary course:** [Kaggle Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

- [ ] Complete Kaggle: *How Models Work*
- [ ] Complete Kaggle: *Basic Data Exploration*
- [ ] Complete Kaggle: *Your First Machine Learning Model*
- [ ] Complete Kaggle: *Model Validation*
- [ ] Complete Kaggle: *Underfitting and Overfitting*
- [ ] Complete Kaggle: *Random Forests*
- [ ] Explain the difference between features, labels, training data, validation data, and test data.
- [ ] Explain underfitting versus overfitting in your own words.
- [ ] In Rust, read a small CSV file and create a train/test split.
- [ ] In Rust, implement mean absolute error (MAE) and mean squared error (MSE).

## Phase 2 — Algebra and Functions (Weeks 3–4)

**Outcome:** Read and manipulate the equations behind basic models.

**Primary resource:** [Khan Academy Algebra](https://www.khanacademy.org/math/algebra)

- [ ] Review variables, equations, and rearranging equations.
- [ ] Review functions and function notation.
- [ ] Review graphing lines and slope.
- [ ] Understand `y = mx + c`.
- [ ] Translate `y = mx + c` into ML notation: `prediction = weight * input + bias`.
- [ ] Review exponents, logarithms, and exponential functions.
- [ ] In Rust, calculate predictions from `prediction = weight * x + bias`.
- [ ] In Rust, change weights and biases and observe the prediction changes.

## Phase 3 — Linear Algebra (Weeks 5–7)

**Outcome:** Understand how multiple inputs and neural-network layers are represented.

**Primary resources:**

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), linear-algebra section

- [ ] Understand scalars, vectors, and matrices.
- [ ] Add and scale vectors.
- [ ] Calculate a dot product.
- [ ] Understand matrix dimensions.
- [ ] Multiply matrices by vectors.
- [ ] Multiply matrices by matrices.
- [ ] Explain a matrix as a linear transformation.
- [ ] Explain why a neural-network layer uses `weights * inputs + bias`.
- [ ] In Rust, implement a `Vector` type or vector helper functions.
- [ ] In Rust, implement matrix-vector multiplication.
- [ ] In Rust, implement multiple linear regression:

```text
prediction = w1 * hours + w2 * sleep + w3 * attendance + bias
```

## Phase 4 — Calculus, Gradients, and Optimization (Weeks 8–9)

**Outcome:** Understand where gradient-descent update rules come from.

**Primary resources:**

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
- [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), calculus section

- [ ] Understand a derivative as a rate of change / slope.
- [ ] Differentiate simple polynomial functions.
- [ ] Understand a partial derivative.
- [ ] Understand a gradient as one partial derivative per parameter.
- [ ] Understand the chain rule conceptually.
- [ ] Explain why optimization moves opposite the gradient.
- [ ] Explain the effect of a learning rate that is too small or too large.
- [ ] Implement MSE loss in Rust.
- [ ] Implement batch gradient descent for single-input linear regression in Rust.
- [ ] Plot or print loss at regular intervals and confirm that it falls during training.
- [ ] Experiment with learning rates `0.001`, `0.01`, `0.1`, and `1.0`.

## Phase 5 — Supervised Learning Foundations (Weeks 10–12)

**Outcome:** Build and evaluate regression and binary-classification models.

**Primary course:** [Machine Learning Specialization — Course 1](https://www.coursera.org/learn/machine-learning)

- [ ] Learn multiple linear regression.
- [ ] Learn feature scaling.
- [ ] Learn logistic regression.
- [ ] Understand the sigmoid function.
- [ ] Understand probability outputs for a binary classifier.
- [ ] Learn binary cross-entropy loss.
- [ ] Understand regularization at a high level.
- [ ] In Rust, implement sigmoid.
- [ ] In Rust, implement logistic regression with gradient descent.
- [ ] In Rust, classify a small binary dataset.
- [ ] Calculate accuracy, precision, recall, and a confusion matrix.

## Phase 6 — Probability and Statistics (Weeks 13–14)

**Outcome:** Understand uncertainty, data variation, and model evaluation more rigorously.

**Primary course:** [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/specializations/mathematics-for-machine-learning-and-data-science), probability and statistics section

- [ ] Calculate mean, median, variance, and standard deviation.
- [ ] Understand normal (Gaussian) distributions.
- [ ] Understand conditional probability.
- [ ] Learn Bayes' rule conceptually.
- [ ] Understand sampling and estimation.
- [ ] Understand the difference between correlation and causation.
- [ ] In Rust, calculate mean and standard deviation for a dataset.
- [ ] In Rust, generate samples from a simple distribution.

## Phase 7 — Classical Machine Learning (Weeks 15–16)

**Outcome:** Understand when a linear model is insufficient and how alternative models behave.

**Primary resources:**

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- [Kaggle Learn](https://www.kaggle.com/learn/)

- [ ] Learn decision trees.
- [ ] Learn random forests.
- [ ] Learn k-nearest neighbors.
- [ ] Learn k-means clustering.
- [ ] Learn the bias–variance tradeoff.
- [ ] Learn basic feature engineering.
- [ ] Train a decision tree using a standard library.
- [ ] Implement either k-nearest neighbors or a small decision tree in Rust.
- [ ] Compare a linear model with a tree-based model on one dataset.

## Phase 8 — Neural Networks (Weeks 17–19)

**Outcome:** Understand and implement the core mechanics of a small neural network.

**Primary course:** [Kaggle Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)

- [ ] Understand a neuron: `activation(weights * inputs + bias)`.
- [ ] Learn ReLU, sigmoid, and softmax activation functions.
- [ ] Understand hidden layers.
- [ ] Implement a forward pass through a two-layer network.
- [ ] Understand backpropagation conceptually.
- [ ] Understand stochastic gradient descent and mini-batches.
- [ ] Learn dropout and batch normalization at a high level.
- [ ] Implement a tiny two-layer neural network in Rust.
- [ ] Train it on a small, inspectable dataset such as XOR.
- [ ] Explain why a single linear model cannot solve XOR.

## Phase 9 — Practical Deep Learning (Week 20 and beyond)

**Outcome:** Train useful modern models with standard tools.

**Primary course:** [fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/)

- [ ] Learn basic PyTorch tensor operations.
- [ ] Train a PyTorch model for an image, text, or tabular-data task.
- [ ] Learn the roles of datasets, dataloaders, models, loss functions, and optimizers.
- [ ] Learn transfer learning.
- [ ] Build one end-to-end project using a public dataset.
- [ ] Read about a Rust ML framework such as Candle or Burn when you are comfortable with the concepts.

## Project milestones

- [ ] **Project 1:** Linear regression from scratch in Rust.
- [ ] **Project 2:** Multi-feature house-price or exam-score predictor in Rust.
- [ ] **Project 3:** Logistic-regression binary classifier in Rust.
- [ ] **Project 4:** Two-layer neural network in Rust.
- [ ] **Project 5:** A practical PyTorch model using a real dataset.

## What to postpone

These are useful in specialized areas, but they are not required for the curriculum above:

- Differential equations
- Partial differential equations (PDEs)
- Abstract algebra
- Rigorous real analysis
- Advanced optimization theory

## Study rule

For every new model, follow this loop:

```text
Understand the equation
  -> implement a small version from scratch
  -> test it with tiny data you can inspect manually
  -> use a library implementation
  -> build a small project
```
