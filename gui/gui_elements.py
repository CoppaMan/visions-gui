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

from models.backend import PiramidVisions, FourierVisions, CLIPCPPN


class VisionGUI:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Starting VisionGUI')
        self.window = tk.Tk()
        self.window.configure(bg='#222222')
        self.window.title("Visions GUI")
        self.window.iconbitmap('icon.ico')
        try:
            self.window.attributes('-type', 'dialog')
        except:
            pass
        self.backend = FourierVisions
        self.model = None
        self.model_done = True

        self.image_viewer = ImageViewer(self)
        self.prompt_bar = PromptBar(self)
        self.model_progress = ModelProgress(self)
        self.settings_panel = SettingsPanel(self)

        self.image_viewer.pack(side='top', expand=True, fill='x')
        self.prompt_bar.pack(side='top', expand=True, fill='x')
        self.model_progress.pack(side='top', expand=True, fill='x')
        self.settings_panel.pack(side='top', expand=True, fill='x')

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

        if not self.source_window.model_done:
            self.after(2000, self.update_image)


class PromptBar(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.logger = logging.getLogger(self.__class__.__name__)

        #self.browse = Button(self, text="AI Art", font='Arial 12 bold', bg='#55dd00', borderwidth=0, highlightthickness=0, command=self.toggle_art)
        self.text_entry = Entry(self, font='Arial 12 bold', borderwidth=0, highlightthickness=0, insertbackground='#dddddd', fg='#dddddd', bg='#666666', justify='center')
        self.button = Button(self, text="GO", font='Arial 12 bold', width=5, borderwidth=0, highlightthickness=0, command=self.start_generation, fg='#dddddd')
        self.set_ready()

        self.button.pack(side='right')
        self.text_entry.pack(side='right', expand=True, fill='both', padx=5, pady=5)
        self.text_entry.grid_rowconfigure(0, weight=1)
        #self.browse.pack(side='right')

    def start_generation(self):
        if len(self.get_prompt()) == 0:
            self.set_error()
            self.logger.error('Prompt cannot be empty')
            return
        else:
            self.set_running()

        if self.source_window.model:
            self.logger.info('Shutting down existing model')
            self.source_window.model.stop()

        self.logger.debug('Loading settings')
        settings = self.source_window.settings_panel.get_settings()
        print(settings)
        
        self.logger.debug('Instanciate new model')
        self.source_window.model = settings['backend']()

        self.source_window.model.set_texts(self.get_prompt())
        self.source_window.model.set_image_detail(settings['cycles_s1'], settings['cycles_s2'])
        self.source_window.model.set_seed(settings['seed'])
        self.source_window.model.set_save_interval(settings['save_every'])

        self.source_window.model.start()
        self.source_window.model_progress.start(settings['cycles_s1'], settings['cycles_s2'])
        self.source_window.image_viewer.start('images/' + self.get_prompt().replace(' ', '_') + '.png')

    def set_ready(self):
        self.button.configure(bg='#679c2a', activebackground='#679c2a', activeforeground='#dddddd', state='normal')

    def set_running(self):
        self.button.configure(bg='#9c962a', activebackground='#9c962a', activeforeground='#dddddd')

    def set_error(self):
        self.button.configure(bg='#9c2a2a', activebackground='#9c2a2a', activeforeground='#dddddd', state='disabled')
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

        self.stage_one = Progressbar(self, orient='horizontal', mode='determinate')
        self.stage_two = Progressbar(self, orient='horizontal', mode='determinate')

        self.stage_one.pack(side='left', expand=True, fill='both', pady=5)
        self.stage_one.grid_rowconfigure(0, weight=1)

        self.stage_two.pack(side='left', expand=True, fill='both', pady=5)
        self.stage_two.grid_rowconfigure(0, weight=1)

    def start(self, cycles_s1, cycles_s2):
        self.cycles_s1 = cycles_s1
        self.cycles_s2 = cycles_s2
        self.stage_one.configure(maximum = cycles_s1)
        self.stage_two.configure(maximum = cycles_s2)

        self.source_window.model_done = False
        self.update_progress()

    def update_progress(self):
        self.logger.debug('Updating progress bar')
        if self.source_window.model is not None:
            progress = self.source_window.model.progress
        else:
            progress = [0, 0]
        self.stage_one['value'] = progress[0]
        self.stage_two['value'] = progress[1]

        if progress[1] < self.cycles_s2:
            self.after(1000, self.update_progress)
        else:
            self.source_window.model_done = True
            self.source_window.prompt_bar.set_ready()


class SettingsPanel(tk.Frame):
    def __init__(self, source_window):
        tk.Frame.__init__(self, source_window.window, bg='#222222')
        self.source_window = source_window

        self.options = [
            #Slider(self, 'Scale', 'max_dim', 128, 2048, steps=128, default=512),
            Slider(self, 'Save every', 'save_every', 20, 1000, steps=20, default=200),
            Slider(self, 'Rough cycles', 'cycles_s1', 100, 5000, steps=100, default=1000),
            Slider(self, 'Fine cycles', 'cycles_s2', 100, 5000, steps=100, default=1000),
            #Slider(self, 'Saturation', 'chroma_noise_scale', -2, 2, steps=.4, default=0),
            #Slider(self, 'Brightness', 'luma_noise_mean', -3, 3, steps=.6, default=0),
            #Slider(self, 'Contrast', 'luma_noise_scale', -2, 2, steps=.4, default=0),
            SeedControl(self),
            Selector(
                self,
                'Backend',
                'backend',
                [
                    ('PyramidVisions', PiramidVisions),
                    ('FourierVisions', FourierVisions),
                    ('CLIP+CPPN', CLIPCPPN)
                ]
            )
        ]

        for option in self.options:
            option.pack(side='top', expand=True, fill='x')

    def get_settings(self):
        result = {}
        for option in self.options:
            result.update(option.get())
        return result


class SeedControl(tk.Frame):
    def __init__(self, frame):
        tk.Frame.__init__(self, frame, bg='#222222')
        self.is_random = True

        self.name_label = Label(self, text='Seed', width=12, font='Arial 12', bg='#222222', fg='#dddddd')
        self.random_seed = Button(
            self, text='Random',
            font='Arial 12',
            fg='#dddddd',
            borderwidth=0,
            highlightthickness=0,
            command=lambda x=True: self.set_seed(x)
        )
        self.fixed_seed = Button(
            self, text='Fixed',
            font='Arial 12',
            fg='#dddddd',
            borderwidth=0,
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

    def get(self):
        seed = None if self.is_random else int(self.seed_entry.get())
        return {'seed': seed}


class Slider(tk.Frame):
    def __init__(self, frame, show_name, var_name, start, end, steps=1, default=None):
        tk.Frame.__init__(self, frame, bg='#222222')
        self.start = start
        self.end = end

        self.var_name = var_name

        self.name_label = Label(self, text=show_name, width=12, font='Arial 12', bg='#222222', fg='#dddddd', justify='left')
        self.start_label = Label(self, text=str(start), font='Arial 8', bg='#222222', fg='#dddddd', width=5)
        self.slider = Scale(self, from_=start, to=end, resolution=steps, orient='horizontal', borderwidth=0, highlightthickness=0, bg='#222222', fg='#dddddd')
        self.slider.set(default if default is not None else start)
        self.end_label = Label(self, text=str(end), font='Arial 8', bg='#222222', fg='#dddddd', width=5)

        self.name_label.pack(side='left')
        self.start_label.pack(side='left')
        self.slider.pack(side='left', expand=True, fill='both')
        self.slider.grid_rowconfigure(0, weight=1)
        self.end_label.pack(side='left')

    def get(self):
        return {
            self.var_name: self.slider.get()
        }


class Selector(tk.Frame):
    def __init__(self, frame, show_name, var_name, options, default=None):
        tk.Frame.__init__(self, frame, bg='#222222')
        self.var_name = var_name

        self.show_names = [ option[0] for option in options ]
        self.values = [ option[1] for option in options ]

        self.current_val = None

        self.name_label = Label(self, text=show_name, width=12, font='Arial 12', bg='#222222', fg='#dddddd')
        self.buttons = [
            Button(
                self, text=self.show_names[button_id],
                font='Arial 12',
                fg='#dddddd',
                borderwidth=0,
                highlightthickness=0,
                command=lambda x=button_id: self.select(x)
            ) for button_id, _ in enumerate(options)
        ]

        self.name_label.pack(side='left')
        for button in self.buttons:
            button.pack(side='left', expand=True, fill='both')
            button.grid_rowconfigure(0, weight=1)

        self.select(0)

    def select(self, button_id):
        self.current_val = self.values[button_id]

        for this_id, button in enumerate(self.buttons):
            if this_id == button_id:
                button.configure(bg='#5a2a9c', activebackground='#5a2a9c', activeforeground='#dddddd')
            else:
                button.configure(bg='#444444', activebackground='#555555', activeforeground='#dddddd')

    def get(self):
        return {
            self.var_name: self.current_val
        }
