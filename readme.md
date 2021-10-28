# Visions GUI

An offline python frontend for the [QuadVisions Colab Notebook](https://colab.research.google.com/drive/1qgMT4-_kDIgZnNGMmrxmwzT3N6Ittw6B?usp=sharing#scrollTo=OOd34BtkuK63) using [tkinter](https://docs.python.org/3/library/tkinter.html).
It offers basic options and interactively displays the generating image.

## Installation
VisionsGUI was tested under Arch Linux and Windows 10 on a NVIDIA GPU. PyramidVisions and FourierVisions require at a card with at least 4GB of VRAM, CLIP+CPPN requires more than 8GB. CPU only is currently not supported.
### Dependencies
Python and CUDA are required, the remaining dependencies can be installed via pip. Using a virtual environment like [venv](https://docs.python.org/3/library/venv.html) is preferred to keep the required pip packages seperate from your existing pip packages.
#### Arch Linux
1) Install Python 3 and pip via `pacman -S python python-pip`
2) Install the CUDA 11 toolkit via `pacman -S cuda`
3) Install the pip requirements: `pip install -r requirements.txt`

#### Windows
1) Install [Python 3](https://www.python.org/downloads/windows/)
2) Install the [CUDA 11 toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
3) Install the pip requirements: `pip.exe install -r requirements.txt`

#### Other Linux distros
Python 3 should be available as a package for most distros but CUDA 11 might not be. In the latter case, an installer is available on [NVIDIA's site](https://developer.nvidia.com/cuda-downloads?target_os=Linux).


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
