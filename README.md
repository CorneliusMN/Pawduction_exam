
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
│
└── source                           <- Source code for the project
    │
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

INSERT CAN BE RUN TO WAY YADAYADA

## Github workflow

INSERT HOW TO RUN THROUGH GITHUB WORKFLOW

## MANUAL

INSERT HOW TO RUN MANUALLY


