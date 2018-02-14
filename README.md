# Anomaly Detector

A project that intends to capture arbitrary data sets, evaluate them against different detectors, select the most appropriate and evaluate when new observations are anomalous.

This project is currently under development and taking its very first steps. Features will be documented in the future.

## How to get started

### Prerequisites

- [Anaconda 5](https://www.anaconda.com/)

### Steps

- Clone this repository.

    ```sh
    git clone https://github.com/makingsensetraining/anomaly-detector.git
    cd anomaly-detector
    ```

- Create a conda environment with the dependencies already provided for you.

    ```sh
    conda env create -f anomaly-detector.yml
    ```

    However, if you already had the environment and you just need to update the dependencies, run:

    ```sh
    conda env update -f anomaly-detector.yml
    ```

- Activate the environment and get hacking!

    ```sh
    activate anomaly-detector
    ```

## Tests

In order to run unit tests, execute in the command line:

```sh
runTests
```