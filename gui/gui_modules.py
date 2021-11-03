import os
import time
import glob
import subprocess
import threading
import logging

import tkinter as tk
from tkinter import Button, Entry, Scale, Label
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image, UnidentifiedImageError

from gui.model_settings import SettingsPanel, PyramidSettings, FourierSettings

from gui.gui_elements import Slider, Selector


class VisionGUI:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Starting VisionGUI')
        self.window = tk.Tk()
        self.window.configure(bg='#222222')
        self.window.title("Visions GUI")
        try:
            self.window.iconbitmap('icon.ico')
        except:
            self.logger.error('WM not supporting icons')
        try:
            self.window.attributes('-type', 'dialog')
        except:
            self.logger.error('WM not supporting floating window')
        self.model = None

        self.image_viewer = ImageViewer(self)
        self.prompt_bar = PromptBar(self)
        self.model_progress = ModelProgress(self)
        self.settings_panel = SettingsManager(self)
        self.seed_control = SeedControl(self)
        self.backend_selector = BackendSelector(self)

        self.image_viewer.pack(side='top', expand=True, fill='x')
        self.prompt_bar.pack(side='top', expand=True, fill='x')
        self.model_progress.pack(side='top', expand=True, fill='x')
        self.settings_panel.pack(side='top', expand=True, fill='x')
        self.seed_control.pack(side='top', expand=True, fill='x')
        self.backend_selector.pack(side='top', expand=True, fill='x')

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show(self):
        self.window.mainloop()

    def on_closing(self):
        if self.model is not None:
            self.model.stop()

        self.window.destroy()


class ImageViewer(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window
        self.path = 'placeholder.jpg'

        self.logger = logging.getLogger(self.__class__.__name__)

        self.img = ImageTk.PhotoImage(Image.open('placeholder.jpg'))
        self.tkimg = tk.Label(self, image = self.img, borderwidth=0, highlightthickness=0, fg='#dddddd')
        self.tkimg.photo = self.img
        self.tkimg.pack()

    def start(self, path):
        self.logger.debug('Image path set to %s', path)
        self.path = path

        self.update_image()

    def update_image(self):
        try:
            self.logger.debug('Refreshing image at %s', self.path)
            self.img = ImageTk.PhotoImage(Image.open(self.path))
            self.tkimg.configure(image=self.img)
            self.tkimg.image = self.img
        except Exception as e:
            self.logger.error('Cannot load image: %s', e)

        if self.source_window.model is not None:
            self.after(2000, self.update_image)


class PromptBar(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = threading.Lock()

        #self.browse = Button(self, text="AI Art", font='Arial 12 bold', bg='#55dd00', borderwidth=0, highlightthickness=0, command=self.toggle_art)
        self.text_entry = Entry(self, font='Arial 12 bold', borderwidth=0, highlightthickness=0, insertbackground='#dddddd', fg='#dddddd', bg='#666666', justify='center')
        self.button = Button(self, text="Start", font='Arial 12 bold', width=5, borderwidth=0, highlightthickness=0, command=self.activate, fg='#dddddd')
        self.set_ready()

        self.button.pack(side='right')
        self.text_entry.pack(side='right', expand=True, fill='both', padx=5, pady=5)
        self.text_entry.grid_rowconfigure(0, weight=1)
        #self.browse.pack(side='right')

    def activate(self):
        with self.lock:
            if self.source_window.model is None:            
                if len(self.get_prompt()) == 0:
                    self.set_error_empty()
                    self.logger.error('Prompt cannot be empty')
                    return

                self.set_running()
                self.source_window.window.update()

                self.logger.info('Starting run')
                
                self.logger.debug('Instanciate new model')
                self.source_window.model = self.source_window.backend_selector.get_backend()()

                # Set global model options
                self.source_window.model.set_texts(self.get_prompt())
                self.source_window.model.set_seed(self.source_window.seed_control.get_seed())

                # Model specific options
                self.source_window.settings_panel.apply_settings(self.source_window.model)

                self.source_window.model.start()
                self.source_window.window.title("Visions GUI - " + self.get_prompt())

                # Start image and progress bar updates
                self.source_window.model_progress.start(self.source_window.settings_panel.get_stages())
                self.source_window.image_viewer.start('images/' + self.get_prompt().replace(' ', '_') + '.png')
            
            else:
                self.logger.info('Stopping run')
                
                self.source_window.model.stop()
                self.source_window.model = None
                self.set_ready()

    def set_ready(self):
        self.button.configure(text='Start', bg='#679c2a', activebackground='#679c2a', activeforeground='#dddddd', state='normal')

    def set_running(self):
        self.button.configure(text='Stop', bg='#9c962a', activebackground='#9c962a', activeforeground='#dddddd')

    def set_error_empty(self):
        self.button.configure(text='Empty', bg='#9c2a2a', activebackground='#9c2a2a', activeforeground='#dddddd', state='disabled')
        self.after(1000, self.set_ready)

    def get_prompt(self):
        return self.text_entry.get()
    
    def toggle_art(self):
        pass


class ModelProgress(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.logger = logging.getLogger(self.__class__.__name__)

        self.stages = []
        self.bars = []

    def set_stages(self, num_cycles):
        pass

    def start(self, stages):
        for bar in self.bars:
            bar.grid_forget()

        self.stages = stages
        self.bars = [ Progressbar(self, maximum=stage, orient='horizontal', mode='determinate') for stage in stages ]
        for n, bar in enumerate(self.bars):

            padding = [1,1]
            if n == 0:
                padding[0] = 5
            if n == len(self.bars)-1:
                padding[1] = 5

            bar.grid(row=0, column=n, sticky='ew', padx=tuple(padding))
            self.grid_columnconfigure(n, weight=stages[n])

        self.update_progress()

    def update_progress(self):
        self.logger.debug('Updating progress bar')
        
        progress = self.source_window.model.progress
        for n, bar in enumerate(self.bars):
            bar['value'] = progress[n]

        if progress[-1] < self.stages[-1] and self.source_window.model is not None:
            self.after(1000, self.update_progress)
        else:
            if self.source_window.model is not None:
                self.source_window.model = None
            self.source_window.prompt_bar.set_ready()


class SettingsManager():
    def __init__(self, source_window):
        self.frame = tk.Frame(source_window.window)
        self.source_window = source_window

        self.logger = logging.getLogger(self.__class__.__name__)

        self.selected = None

        self.panels = {
            settings_classes.backend: settings_classes(self.frame) for settings_classes in SettingsPanel.__subclasses__()
        }

        self.logger.info('Loaded settings: %s', self.panels)

    def pack(self, **kwargs):
        self.frame.pack(kwargs)

    def select(self, backend):
        if self.selected is not None:
            self.selected.pack_forget()
        self.selected = self.panels[backend]
        self.selected.pack(side='top', expand=True, fill='x')

    def get_stages(self):
        return self.selected.get_settings()['cycles']

    def apply_settings(self, model):
        return self.selected.apply_options(model)
        

class SeedControl(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.is_random = True

        self.name_label = Label(self, text='Seed', width=12, font='Arial 12', bg='#222222', fg='#dddddd')
        self.random_seed = Button(
            self, text='Random',
            font='Arial 12',
            fg='#dddddd',
            borderwidth=0,
            width=8,
            highlightthickness=0,
            command=lambda x=True: self.set_seed(x)
        )
        self.fixed_seed = Button(
            self, text='Fixed',
            font='Arial 12',
            fg='#dddddd',
            borderwidth=0,
            width=8,
            highlightthickness=0,
            command=lambda x=False: self.set_seed(x)
        )
        self.seed_entry = Entry(self, font='Arial 12', borderwidth=0, highlightthickness=0, insertbackground='#dddddd', fg='#dddddd', bg='#444444', justify='right')

        self.name_label.pack(side='left')
        self.random_seed.pack(side='left')
        self.fixed_seed.pack(side='left')
        self.seed_entry.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        self.seed_entry.grid_rowconfigure(0, weight=1)

        self.set_seed(True)

    def set_seed(self, is_random):
        if is_random:
            self.is_random = True
            self.random_seed.configure(bg='#5a2a9c', activebackground='#5a2a9c', activeforeground='#dddddd')
            self.fixed_seed.configure(bg='#444444', activebackground='#555555', activeforeground='#dddddd')
        else:
            self.is_random = False
            self.random_seed.configure(bg='#444444', activebackground='#555555', activeforeground='#dddddd')
            self.fixed_seed.configure(bg='#5a2a9c', activebackground='#5a2a9c', activeforeground='#dddddd')

    def get_seed(self):
        seed = None if self.is_random else int(self.seed_entry.get())
        return seed


class BackendSelector(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.backend_selector = Selector(
            self,
            'Backend',
            'backend',
            [
                (backend.__name__, backend) for backend in self.source_window.settings_panel.panels
            ],
            command = self.update_settings
        )

        self.backend_selector.pack(side='left', expand=True, fill='both')
        self.backend_selector.grid_rowconfigure(0, weight=1)

    def update_settings(self, value):
        self.source_window.settings_panel.select(value)

    def get_backend(self):
        return self.backend_selector.get()['backend']