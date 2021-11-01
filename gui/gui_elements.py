'''
Additional Tkinter objects can be defined here
'''

import tkinter as tk
from tkinter import Button, Entry, Scale, Label

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
    def __init__(self, frame, show_name, var_name, options, default=None, command=None):
        tk.Frame.__init__(self, frame, bg='#222222')
        self.var_name = var_name
        self.command = command

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

        if self.command is not None:
            self.command(self.current_val)

    def get(self):
        return {
            self.var_name: self.current_val
        }
