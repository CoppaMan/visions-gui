import os
import re
import time
import subprocess
import threading
import glob

import tkinter as tk
from tkinter import Frame, Button, Entry, Toplevel, Scrollbar
from PIL import ImageTk, Image, ImageFile


class ArtWindow:
    def __init__(self, master):
        self.master = master

        self.art_window = Toplevel(self.master.window)
        self.art_window.title('Gallerie')
        self.art_window.attributes('-type', 'dialog')
        self.art_window.withdraw()
        self.art_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.visible = False

        self.art_panel = ArtPanel(self)
        self.art_panel.place()

        #self.scroll = Scrollbar(self.art_window, orient='horizontal')
        #self.scroll.pack(side='bottom')
        #self.scroll.config(command=self.art_panel.xview)

    def on_closing(self):
        self.hide()

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()

    def show(self):
        self.art_window.deiconify()
        self.visible = True

    def hide(self):
        self.art_window.withdraw()
        self.visible = False

class ArtPanel:
    def __init__(self, master):
        self.frame = Frame(master.art_window, width=1000)
        self.match = re.compile(r'\d{6}')

        self.images = []
        self.image_name = []
        self.scan_images()
        self.show_images()

    def scan_images(self):
        for image in glob.glob('images/*'):
            if os.path.isdir(image):
                continue
            if re.search(self.match, image):
                os.remove(image)
                continue

            self.image_name.append(image)

    def show_images(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for image in self.images:
            image.pack_forget()

        for name in self.image_name:
            img = ImageTk.PhotoImage(Image.open(name))
            picture = tk.Label(self.frame, image = img, borderwidth=0, highlightthickness=0)
            picture.photo = img
            self.images.append(picture)
            picture.pack(side='right')

        ImageFile.LOAD_TRUNCATED_IMAGES = False

    def place(self):
        self.frame.pack(side='top')