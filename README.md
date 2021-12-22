# ML4Science Project: The Epoch of Reionisation :milky_way:
Repository for the second project of the course CS-433 Machine Learning @ EPFL. 
The team is composed by:
- *Matteo Calafà* ([@Teocala](https://github.com/Teocala))
- *Giulia Mescolini* ([@giuliamesc](https://github.com/giuliamesc)) 
- *Paolo Motta* ([@paolomotta](https://github.com/paolomotta))

We have worked in the framework of *ML4Science* projects and collaborated with the *LASTRO* (Laboratory of Astrophysics) of EPFL Lausanne, under the supervision of Dr. Michele Bianco ([@micbia](https://github.com/micbia)).

The aim of the project is to enhance Machine Learning usage in the study of the radiation behavior in the universe during the Epoch of Reionisation, that is, the period of formation of the first galaxies and stars.

## Data Loading
- Download the data from [here](https://drive.google.com/drive/folders/1d-FjkS6f8e1Q5F3k0Yz2rygxk8f7hqS_?usp=sharing)
- Store them in a folder called `dataset`, on the same level of your local `reionisation_ML` repository
- Run the script `neigh_generation.py`, which will generate a folder `cubes` on the same level of your code, containing the input data (which consists of neighborhoods of points) for the NN

## Packages Needed
We have projected our Neural Network with `torch`, version (INSERT PAOLO'S VERSION).

The packages that are needed for the project are:
### Core
- `numpy`
- `matplotlib`
- `torch`
- `sklearn`
### Utilities
- `pickle`
- `gc`
- `time `



## Structure
The code structure is the following:
- `main.py`, the Python script to run
- `FNN.py`, importable script containing the definition of the Fully Connected Neural Network
- `CNN.py`, importable script containing the definition of the Convolutional Neural Network
- `neigh_generation.py`, which preprocesses the input for the CNN
- `parameters.py`, which contains a list of parameters that you can set here without modifying the main each time (eg.: batch size, number of epochs)
- `plotting.py`, which generates some plots used for accuracy evaluation of the NNs 
- `model_55.pt`, is the best CNN model obtained (corresponding to epoch 55)
- Folder `Weekly Meetings` contains our presentations to the tutor with the updates of our work of the week
- `Report.pdf` is the final 4-pages report delivered

## Instructions for training
Our code contains a feature which enables to stop the training and then to restart it from the point on which we interrupted it; thanks to this strategy, the net can be trained on a laptop without being forced to wait until a very long training is completed, and at the same time ensuring a backup in case something goes wrong. 
- In the first epoch, the status of the net, the losses and the R² score are saved
- In the following epochs, the above information are stored 1) for the best model 2) for the last epoch, in order to continue from it the next time we restart the training

All these files are stored in the folder `model`.
