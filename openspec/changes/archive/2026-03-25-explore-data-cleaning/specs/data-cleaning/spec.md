## ADDED Requirements

### Requirement: Load and validate raw CSV data
The system SHALL load `ml_challenge_dataset.csv` and validate that it contains the expected columns and three painting labels.

#### Scenario: Successful load
- **WHEN** the script reads the CSV file
- **THEN** the script SHALL report the total row count, column names, and the count per painting label

#### Scenario: Missing value report
- **WHEN** the data is loaded
- **THEN** the script SHALL print a per-column missing value count and percentage

### Requirement: Clean numeric features
The system SHALL extract numeric values from the 7 core numeric/ordinal columns: `emotion_intensity`, `sombre`, `content`, `calm`, `uneasy`, `num_colours`, `num_objects`.

#### Scenario: Ordinal Likert extraction
- **WHEN** a value is in the format `"4 - Agree"`
- **THEN** the system SHALL extract the leading integer `4`

#### Scenario: Handle non-numeric entries
- **WHEN** a numeric column contains a non-parseable value (e.g., empty string, text)
- **THEN** the system SHALL set it to NaN and fill with the column median

### Requirement: Clean willing_to_pay
The system SHALL extract a numeric dollar amount from the free-text `willing_to_pay` column.

#### Scenario: Standard dollar formats
- **WHEN** the value is `"$50"`, `"100,000"`, `"200 dollars"`, or `"500 CAD"`
- **THEN** the system SHALL extract the numeric portion as a float

#### Scenario: Unparseable values
- **WHEN** the value cannot be parsed (e.g., `"I wouldn't pay"`, `"a"`)
- **THEN** the system SHALL set it to NaN and fill with the column median

### Requirement: One-Hot encode multi-select columns
The system SHALL one-hot encode the three multi-select columns: `room`, `view_with`, `season`.

#### Scenario: Multi-value cell
- **WHEN** a cell contains `"Office,Living room"`
- **THEN** the system SHALL produce `room_Office=1` and `room_Living room=1`, with all other room columns set to 0

#### Scenario: Missing multi-select
- **WHEN** a multi-select cell is empty
- **THEN** all corresponding one-hot columns SHALL be set to 0

### Requirement: Drop rows with no usable features
The system SHALL drop any row where all feature columns (excluding `unique_id` and `Painting`) are empty.

#### Scenario: Completely empty response
- **WHEN** a row has only `unique_id` and `Painting` filled
- **THEN** the row SHALL be removed from the dataset

### Requirement: Standardize numeric features
The system SHALL apply StandardScaler to all numeric feature columns and save the scaler parameters (mean, std) for later use.

#### Scenario: Scaler fit on training data
- **WHEN** the training split is created
- **THEN** the scaler SHALL be fit on training data only and applied (transform) to both train and validation data

### Requirement: Split dataset
The system SHALL split the cleaned data into train (80%) and validation (20%) sets using stratified sampling.

#### Scenario: Stratified split
- **WHEN** the data is split
- **THEN** each split SHALL maintain the same painting label proportions as the full dataset

#### Scenario: Reproducibility
- **WHEN** the split is performed
- **THEN** a fixed `random_state=42` SHALL be used for reproducibility
