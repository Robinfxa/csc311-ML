## ADDED Requirements

### Requirement: Pure numpy softmax inference
The system SHALL implement logistic regression inference using only numpy, without sklearn.

#### Scenario: Forward pass
- **WHEN** given a feature matrix X, weights W, and bias b
- **THEN** the system SHALL compute `predictions = argmax(X @ W.T + b)` and map indices back to painting names

### Requirement: Consistency verification
The system SHALL verify that numpy predictions match sklearn predictions on the validation set.

#### Scenario: Prediction match
- **WHEN** both sklearn and numpy predictions are produced for X_val
- **THEN** they SHALL be identical for every sample

### Requirement: Standalone predict_all function
The system SHALL provide a `predict_all(filename)` function that takes a CSV path and returns a list of painting name predictions.

#### Scenario: CSV input prediction
- **WHEN** `predict_all("test.csv")` is called
- **THEN** it SHALL load the CSV, apply the same preprocessing (ordinal extraction, numeric cleaning, one-hot encoding, scaling), and return predictions as a list of painting name strings
