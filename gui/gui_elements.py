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


class WeightedPrompt(tk.Frame):
    def __init__(self, frame):
        tk.Frame.__init__(self, frame, bg='#222222')

        self.prompt = Entry(self, font='Arial 12 bold', borderwidth=0, highlightthickness=0, insertbackground='#dddddd', fg='#dddddd', bg='#666666', justify='center')
        self.slider = Scale(self, from_=-1, to=1, length=100, resolution=.2, orient='horizontal', borderwidth=0, highlightthickness=0, bg='#222222', fg='#dddddd')
        self.slider.set(0)

        self.prompt.pack(side='left', expand=True, fill='x', padx=(5,5))
        self.prompt.grid_rowconfigure(0, weight=1)
        self.slider.pack(side='left')

    def get(self):
        return ( self.prompt.get(), self.slider.get() )

class ExtraPrompts(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root.window, bg='#222222')

        self.name_frame = tk.Frame(self, bg='#222222')
        self.name = Label(self.name_frame, text='Weighted prompts', font='Arial 12', bg='#222222', fg='#dddddd')
        self.add_button = Button(
            self.name_frame,
            text="+",
            font='Arial 10 bold',
            borderwidth=0,
            highlightthickness=0,
            command=self.add_prompt,
            bg='#444444',
            fg='#dddddd',
            activebackground='#666666',
            activeforeground='#dddddd'
        )
        self.remove_button = Button(
            self.name_frame,
            text="-",
            font='Arial 10 bold',
            borderwidth=0,
            highlightthickness=0,
            command=self.remove_prompt,
            bg='#444444',
            fg='#dddddd',
            activebackground='#666666',
            activeforeground='#dddddd'
        )
        self.name.pack(side='top')
        self.add_button.pack(side='left', padx=5, pady=5, expand=True, fill='x')
        self.add_button.grid_rowconfigure(0, weight=1)
        self.remove_button.pack(side='left', padx=5, pady=5, expand=True, fill='x')
        self.remove_button.grid_rowconfigure(0, weight=1)

        self.prompt_frame = tk.Frame(self, bg='#222222')
        self.no_prompt = Label(self.prompt_frame, text='No additional prompts', width=12, font='Arial 12', bg='#222222', fg='#dddddd')
        self.no_prompt.pack(side='left', expand=True, fill='x')
        self.no_prompt.grid_rowconfigure(0, weight=1)

        self.prompts = []

        self.name_frame.pack(side='left', anchor='n')
        self.prompt_frame.pack(side='left', expand=True, fill='both', anchor='n')
        self.prompt_frame.grid_rowconfigure(0, weight=1)

    def add_prompt(self):
        if len(self.prompts) < 5:
            if len(self.prompts) == 0:
                self.no_prompt.pack_forget()

            new_prompt = WeightedPrompt(self.prompt_frame)
            new_prompt.pack(side='top', expand=True, fill='x', anchor='n')
            new_prompt.grid_rowconfigure(0, weight=1)

            self.prompts.append(new_prompt)

    def remove_prompt(self):
        if len(self.prompts) > 0:
            old_prompt = self.prompts.pop()
            old_prompt.pack_forget()

            if len(self.prompts) == 0:
                self.no_prompt.pack(side='left', expand=True, fill='x')
                self.no_prompt.grid_rowconfigure(0, weight=1)

    def get_weighted_prompts(self):
        return [ prompt.get() for prompt in self.prompts ]

