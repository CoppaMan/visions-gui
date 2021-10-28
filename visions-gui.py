import os
import logging
from gui.gui_elements import VisionGUI

logging.basicConfig(level=logging.DEBUG)

if not os.path.exists('images'):
    os.makedirs('images')

root = VisionGUI()
root.show()
