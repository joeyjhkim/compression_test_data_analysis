# Compression Testing Data Processing and Visualization
This repository contains Python code for processing and visualizing compression testing data for soft materials. 
The script automatically reads raw force–displacement CSV data from multiple trials, cleans it, and generates a multi-page PDF report with smooth fitted curves for each material configuration.

## Features
- Automatic File Detection – Reads all .csv files in each material folder chronologically (Trial 1 → Trial 2 → Trial 3).
- Noise Reduction – Averages repeated displacement values to remove local actuator noise and chatter.
- Data Splitting – Separates each dataset into loading and unloading phases based on the maximum displacement point.
- Spline Curve Fitting – Uses cubic spline interpolation with Savitzky–Golay smoothing to create continuous, realistic force–displacement curves.
- Automatic Averaging – Computes average curves across all trials in each material category.
- Yield Marker – Highlights the initial peak (maximum force) on averaged curves with a vertical reference line.
- PDF Report Generation – Exports all results into a clean, publication-ready PDF file.

## Data Processing Summary
1.	Identify the maximum displacement point in each CSV file.
2.	Split the data into two sets:
- Loading phase → before reaching peak displacement
- Unloading phase → after peak displacement
3.	For each set, if multiple rows share the same displacement, average their force values.
4.	Fit a smooth spline curve through the cleaned data for each phase.
5.	Repeat for every trial and average across all trials to create section-level plots.

## Requirements
- Python >= 3.8
- pandas
- numpy
- matplotlib
- scipy


## How to Run
`python3 reportgeneration.py`


