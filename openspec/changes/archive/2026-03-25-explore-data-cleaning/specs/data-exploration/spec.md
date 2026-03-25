## ADDED Requirements

### Requirement: Feature distribution visualization
The system SHALL generate histograms showing the distribution of each numeric feature, grouped by painting label.

#### Scenario: Grouped histogram output
- **WHEN** the EDA is run
- **THEN** the script SHALL save a figure with subplots showing the distribution of each numeric feature per painting class

### Requirement: Correlation analysis
The system SHALL compute and visualize the correlation matrix of all numeric features.

#### Scenario: Heatmap output
- **WHEN** the EDA is run
- **THEN** the script SHALL save a heatmap image of the feature correlation matrix

### Requirement: Class distribution visualization
The system SHALL display the count of each painting label in a bar chart.

#### Scenario: Bar chart output
- **WHEN** the EDA is run
- **THEN** the script SHALL save a bar chart showing the number of samples per painting class
