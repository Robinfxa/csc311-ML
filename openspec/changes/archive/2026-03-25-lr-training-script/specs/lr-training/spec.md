## ADDED Requirements

### Requirement: Load cleaned data
The system SHALL load the standardized train/val splits from `cleaned_data/` and encode painting labels to integers.

#### Scenario: Data loading
- **WHEN** `train_lr.py` is run
- **THEN** it SHALL load X_train.npy, X_val.npy, y_train.npy, y_val.npy and map labels to integers {0, 1, 2}

### Requirement: Grid search over C and penalty
The system SHALL perform a grid search over regularization strength C and penalty type using stratified K-Fold cross-validation.

#### Scenario: Full grid search
- **WHEN** the grid search runs
- **THEN** it SHALL evaluate C ∈ {0.001, 0.01, 0.1, 1, 10, 100} × penalty ∈ {l1, l2} using 5-fold stratified CV and record mean/std accuracy for each combination

#### Scenario: Results saved
- **WHEN** the grid search completes
- **THEN** the system SHALL save a CSV with columns: C, penalty, mean_cv_accuracy, std_cv_accuracy, train_accuracy

### Requirement: Train best model
The system SHALL train a final model using the best hyperparameters found from grid search on the full training set.

#### Scenario: Best model training
- **WHEN** the best C and penalty are identified
- **THEN** the system SHALL re-train on full training data and evaluate on the held-out validation set

### Requirement: Classification report
The system SHALL produce a per-class classification report and confusion matrix for the best model.

#### Scenario: Report output
- **WHEN** the best model is evaluated on the validation set
- **THEN** the system SHALL print precision, recall, F1 per class and overall accuracy

#### Scenario: Confusion matrix visualization
- **WHEN** the best model is evaluated
- **THEN** the system SHALL save a confusion matrix heatmap to `lr_results/`

### Requirement: Hyperparameter tuning visualization
The system SHALL generate a line plot showing validation accuracy vs C for each penalty type.

#### Scenario: Tuning curve plot
- **WHEN** the grid search CSV exists
- **THEN** the script SHALL save a plot with C (log scale) on x-axis and mean CV accuracy on y-axis, with separate lines for L1 and L2

### Requirement: Export model parameters
The system SHALL export the trained model's weights and bias as numpy arrays.

#### Scenario: Parameter export
- **WHEN** the best model is trained
- **THEN** it SHALL save `lr_weights.npy` (shape: n_classes × n_features), `lr_bias.npy` (shape: n_classes,), and `label_map.txt` to `lr_results/`
