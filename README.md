# Introduction to Artificial Intelligence - Projects

This repository contains projects developed as part of the **Introduction to Artificial Intelligence** course at Warsaw University of Technology (WUT), 1st year. Each folder corresponds to a different AI or machine learning topic. You will find both algorithm implementations and PDF reports of each task.

---

## Repository Structure

### DecisionTree
- **Description:** Implementation of a decision tree classifier.
- **Files:**
  - `prepare_data.py` - Data preparation and preprocessing.
  - `evaluate_model.py` - Model evaluation.
  - `main.py` - Final usage of the model.
  - `test_model.py` - Unit tests for the model.
  - **Report:** `report.pdf`

### DotsAndBoxesGame
- **Description:** Implementation of the "Dots and Boxes" game and the Min-Max algorithm for decision-making.
- **Files:**
  - `Game.py` - Game logic.
  - `MinMax.py` - Min-Max algorithm.
  - `State.py` - Game state representation.
  - `test_min_max.py` - Tests for the Min-Max algorithm.
  - **Report:** `report.pdf`

### GeneticAlgorithm
- **Description:** Implementation of a genetic algorithm for optimization.
- **Files:**
  - `evaluate.py` - Function for evaluating the population.
  - `GeneticAlgorithm.py` - Main implementation of the algorithm.
  - `solver.py` - Abstract class solver.
  - `test_genetic.py` - Algorithm tests.
  - **Report:** `report.pdf`

### GradientDescend
- **Description:** Optimization of functions using the gradient descent algorithm.
- **Files:**
  - `config.py` - Configuration file.
  - `GradientDescend.py` - Algorithm implementation.
  - `solver.py` - Abstract class solver.
  - `test_gradient.py` - Tests for gradient descent.
  - **Report:** `report.pdf`

### MLP (Multilayer Perceptron)
- **Description:** Implementation of a multilayer perceptron (MLP) neural network.
- **Files:**
  - `activation_functions.py` - Activation functions for the neural network.
  - `mlp.py` - Implementation of the MLP.
  - `main.py` - Final usage of the model.
  - **Report:** `report.pdf`

### NaiveBayes
- **Description:** Naive Bayes classifier implementation.
- **Files:**
  - `NaiveBayesClassifier.py` - Classifier implementation.
  - `scripts.py` - Helper scripts.
  - `solver.py` - Abstract class solver.
  - `data_exploration.ipynb` - Notebook for data exploration.
  - **Report:** `report.pdf`

### QLearning
- **Description:** Implementation of the Q-learning algorithm for reinforcement learning.
- **Files:**
  - `LearningModel.py` - Implementation of the Q-learning model.
  - `main.py` - Experiments with the model.
  - **Report:** `report.pdf`

---

## Requirements
To run the projects, you need Python 3.8+ and the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
