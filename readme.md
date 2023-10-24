# Melanoma Segmentation Challenge Submission - Group-6

## Contributors:
- Dheeraj Varghese
- Danila R.

## Overview:

The provided code includes the necessary files to replicate the experiments conducted on the Snellius cluster.

## Code Structure:

- **melTransforms:** Holds all the transformations utilized for preprocessing the dataset.
- **models:** Explore the implementation of various models, including Unet, Code-Unet, Resid-Unet, Attention-Unet, MAnet, and Unetplusplus.
- **train:** The main training file that orchestrates the training process.

## Running Experiments:

To execute the experiments on your local environment or cluster, refer to the "experiments.sh" file. Please ensure to install the required libraries specified in the "requirements.txt" before initiating any experiments. Use the following command for installation:

```bash
pip install -r requirements.txt
```

## How to Run:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/ColdCoffee21/melanoma-segmentation-challenge.git
```

2. Navigate to the project directory:

```bash
cd melanoma-segmentation-challenge
```

3. Download the data from [here](https://drive.google.com/drive/folders/1UyQtp3SQg5jejcx-intB3BPr_L4VGXoE?usp=drive_link).

4. Activate your conda environment.

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Execute experiments using the provided script:

```bash
sbatch experiments.sh
```

## Notes:

- The experiments were conducted on the Snellius cluster.
- Ensure that all dependencies are installed before running any experiments.

Feel free to explore and modify the code as needed. If you encounter any issues or have questions, please reach out to the contributors.

Happy coding!
