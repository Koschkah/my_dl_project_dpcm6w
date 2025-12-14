# Project Details
## Project Information
- Selected Topic:  End-of-trip delay prediction
- Student Name: Bakos Máté
- Aiming for +1 Mark: NO

## Solution Description

This project implements a Docker-based machine learning pipeline for predicting public transport vehicle delay at the final stop of a trip. Several baseline models (mean, current delay, and linear regression) are used, alongside a GraphSAGE-based Graph Neural Network that models spatial relationships between transit stops using a stop graph. The models are trained with a time-based data split to prevent temporal leakage. 
- Due to runtime constraints, the training data size is intentionally reduced while preserving the full pipeline and model structure.
 Performance is evaluated using MAE, RMSE, and threshold-based accuracy metrics (60, 120, and 180 seconds). And everything is logged in text file for tracking.

## Extra Credit Justification

No extra credit!

## Docker Instructions
- Build 
```bash
docker build -t my-dl-project-work-app:1.0 .
```
- Run
```bash
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\output:/app/output" my-dl-project-work-app:1.0 > training_log.txt 2>&1
```

## File structure 

**src/:** Contains the source code for the machine learning pipeline.

- 01-data-preprocessing.py: Scripts for loading, cleaning, and preprocessing the raw data.
- 02-training.py: The main script for defining the model and executing the training loop.
- 03-evaluation.py: Scripts for evaluating the trained model on test data and generating metrics.
- data_download.py: Script to download the data from google drive.

**notebook/:** Contains Jupyter notebooks for analysis and experimentation.

- 01_data_processing.ipynb: For experimenting Scripts for loading, cleaning, and preprocessing the raw data.
- 1_batch_overfit.ipynb: To test if MLP can be overfitted on 1 batch
- 02_train.ipynb: For experimenting For experimenting The main script for defining the model and executing the training loop.
- 03_evaluation.ipynb: For experimenting Scripts for evaluating the trained model on test data and generating metrics.
- BKK_API_query: For the data query from the BKK API
- data_eval.ipynb: Initial data testing

**Root Directory:**

- Dockerfile: Configuration file for building the Docker image with the necessary environment and dependencies.
- requirements.txt: List of Python dependencies required for the project.
- README.md: Project documentation and instructions.
- training_log.txt: for saving the logs of the code.

