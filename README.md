
# ITU BDS MLOPS'25 - Project

This project was done as part of the course Data Science in Production: MLOps and Software Engineering (Autumn 2025) at the IT University of Copenhagen. For more information click [here.](https://github.com/lasselundstenjensen/itu-sdse-project)

The project is made up of cookiecutter files, organized with a daggerworkflow in go, which is orchestrated by a github workflow.
![Project architecture](./docs/project-architecture.png)

## Project Structure

```
├── README.md                        <- Project description, project structure, how to run the code
│
├── .dvc/
│   │
│   └── config                       <- Contains reference to remote/remote-URL for dvc pulling
│
├── .github/workflows                <- GitHub workflows
│   │
│   └── train_test_action.yml        <- Workflow that automatically trains and tests model
│
├── data                             <- Data-folder
│   │
│   └── raw_data.csv.dvc             <- DVC file for raw_data
|
├── artifacts                        <- Directory for model artifacts
|
├── mlruns                           <- Directory for MLRun logged info
│
├── docs                             <- Contains graphs used in markdown file
│
├── go                               <- Contains GO files for the project
│   │
│   └── pipeline.go                  <- The dagger workflow in GO
│
├── go.mod                           <- Go file for the module and required dependencies
│
├── go.sum                           <- Go file for continuity and integrity of dependencies
│
├── Makefile                         <- Creates the necesarry project structure for the workflow
│
├── requirements.txt                 <- Python dependencies needed for the project
│
└── source                           <- Source code for the project
    │
    ├── config.py       <- Python script with essential configs like paths
    │
    ├── data.py         <- Script for loading data and filtering
    │
    ├── preprocess.py   <- Script for performing preprocessing steps on data
    │
    ├── train.py        <- Script for training and saving best model
    │
    ├── evaluation.py   <- Script for evaluating best model performance
    │
    ├── util.py         <- Utility functions used by scripts
    │
    └── wrappers.py     <- Wrappers used by training models

```

# How to Run the Project

The project is intended to be run using github workflow, however if you wish it can also be run locally.

## Github workflow

From Actions pane in github run Train and Test ML Model workflow. This will run both the artifact generation and the inference test against the model.
You will be able to find all artifacts produced from the run, under the "Artifacts" folder. Find Linear Regression model under artifacts named "model". 
In the MLRuns folder you will be able to find logged MLFlow data. In the Data folder, you will be able to find the splits for test/train, the gold data etc.

## MANUAL

For manually running the pipeline you will need to set up an environment on your computer. The go pipeline itself will handle the dependencies of the python scripts themselves, but you must have the following to run the code:
'''
dagger >= v0.18.16
go version >= go1.25.0
Python >= 3.11.9
Docker >= 28.3.2
'''
However, older versions might be fine to run as well. When you have the environment setup you should launch docker desktop if using this, and then run the following from your terminal in the root directory run the following:
'''
make
go run pipeline.go
'''
This will pull the data, make directories for model artifacts, data and mlflow logs and produce the model artifacts in a setup similar to if you had run it through github workflow.

