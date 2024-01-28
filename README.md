
# Prevision Library

## Description
Prevision is a Python library designed for advanced data assimilation and analysis. It provides a robust framework for integrating various mathematical and statistical models, particularly focusing on the Ensemble Kalman Filter (EnKF) technique.

## Code Architecture
The library is structured around key classes and functions:
- `Data_Assimilation`: An abstract base class that lays the groundwork for data assimilation methods.
- Subclasses of `Data_Assimilation`: These are likely to include specific implementations of data assimilation techniques like EnKF.
- Utility functions and classes for additional processing and analysis, including Fourier Neural Operators implementation.

## Repository Structure:
- `data` folder contains all data generated in and used for the examples present in the repository.
- `scripts` floder contains a series of examples of usage of the `prevision` library and the library itself. Please see the files `REPORT_Prevision.pdf` and `SLIDES_Prevision.pdf` for a complete description of the library and the a full guide to the examples. The naming convention for the files in the `scripts` folder follows the naming of the corresponding Appendix sections in `REPORT_Prevision.pdf`

## Dependencies
Prevision relies on several external libraries:
- `numpy`: For numerical operations and handling large arrays and matrices.
- `matplotlib`: Used for plotting and visualizing data.
- `tensorflow` and `keras`: For building and training neural network models.
- `filterpy`: Specifically for Kalman filtering techniques.
- `IPython`: For interactive computing and display features.

Ensure these dependencies are installed in your environment:
```bash
pip install numpy matplotlib tensorflow keras filterpy IPython
```

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/Adrianominora/Prevision
cd [repository-directory]
```

## Usage
Here is a basic example of how to use the `Data_Assimilation` class in Prevision:

```python
from Library import Data_Assimilation

# Example function definitions for f and h
def f(x):
    # Define your state transition function here
    pass

def h(x):
    # Define your measurement function here
    pass

# Creating an instance of a Data_Assimilation subclass
model = Data_Assimilation_Subclass(dim_x, dim_z, f, h, get_data)

# Example usage of the model
model.method_name(args)
```

Replace `Data_Assimilation_Subclass` and `method_name` with the actual subclass and methods you wish to use.

## Authors and Acknowledgment
- [Adriano Minora] - Initial work
- [Giacomo Mondello Malvestiti] - Initial work
- [Stefano Pagani] - Supervision
