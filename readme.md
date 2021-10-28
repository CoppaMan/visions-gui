# Visions GUI

An offline python frontend for the [QuadVisions Colab Notebook](https://colab.research.google.com/drive/1qgMT4-_kDIgZnNGMmrxmwzT3N6Ittw6B?usp=sharing#scrollTo=OOd34BtkuK63) using [tkinter](https://docs.python.org/3/library/tkinter.html).
It offers basic options and interactively displays the generating image.

## Installation

1) Install Python 3
2) Install CUDA
3) Install pip requirements: `pip install -r requirements`


## Usage
To start, run `python visions-gui.py`
![alt text](docs/gui_start.png "The GUI")

### Modules
The GUI is made up of multiple modules which offer feedback to the user and allow to change a handful of settings.
#### Image viewer
The image viewer displays the current progress of the image and is periodically updated during generation. The displayed images are also accessible in the `images` folder.
#### Progress bar
The progress bar shows the progress of the two stages from left to right.
#### Prompt bar
The prompt bar consists of a text field and a button labeled GO. A prompt is entered into the text field and the image will be generated as soon as the button is pressed.
The button also serves as the GUI's state:
|Color |State |
|-|:-|
|Green| The model is ready and is waiting to be started |
|Yellow| The model is currently generating an image |
|Red| An error occured |
#### Settings panel
The settings panel contains a few option for generating an image:
|Setting |Description |
|-|:-|
|Save every| Cycles completed before displaying and saving the next image. Lower values add more overhead. |
|Rough cycles| Cycles during the first stage. |
|Fine cycles| Cycles during the second stage one. |
|Seed| If a random or fixed seed is used. |
|Backend| Which backend to use |
