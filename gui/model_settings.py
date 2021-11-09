from tkinter import Frame

from models.visions import PyramidVisions, DirectVisions, FourierVisions

from gui.gui_elements import Slider


class SettingsPanel(Frame):
    backend = None

    def __init__(self, frame):
        Frame.__init__(self, frame, bg='#222222')
        self.options = {}

    def pack_options(self):
        for option in self.options:
            option.pack(side='top', expand=True, fill='x')

    def get_settings(self):
        result = {}
        for option in self.options:
            for name, value in option.get().items():
                if name in result:
                    if isinstance(result[name], list):
                        result[name].append(value)
                    else:
                        result[name] = [result[name], value]
                else:
                    result[name] = value
        return result


class PyramidSettings(SettingsPanel):
    backend = PyramidVisions

    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Save every', 'save_interval', 20, 1000, steps=20, default=200),
            Slider(self, 'Rough cycles', 'cycles', 100, 5000, steps=100, default=1000),
            Slider(self, 'Fine cycles', 'cycles', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()

    def apply_options(self, model):
        options = self.get_settings()
        model.stages[0]['cycles'] = options['cycles'][0]
        model.stages[1]['cycles'] = options['cycles'][1]
        model.save_interval = options['save_interval']


class DirectSettings(SettingsPanel):
    backend = DirectVisions

    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Save every', 'save_interval', 20, 1000, steps=20, default=100),
            Slider(self, 'Stage 1', 'cycles', 100, 5000, steps=100, default=500),
            Slider(self, 'Stage 2', 'cycles', 100, 5000, steps=100, default=500),
            Slider(self, 'Stage 3', 'cycles', 100, 5000, steps=100, default=500),
            Slider(self, 'Stage 4', 'cycles', 100, 5000, steps=100, default=500),
            Slider(self, 'Stage 5', 'cycles', 100, 5000, steps=100, default=700),
            Slider(self, 'Stage 6', 'cycles', 100, 5000, steps=100, default=1000),
            Slider(self, 'Stage 7', 'cycles', 100, 5000, steps=100, default=1000),
            Slider(self, 'Stage 8', 'cycles', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()

    def apply_options(self, model):
        options = self.get_settings()
        for n in range(8):
            model.stages[n]['cycles'] = options['cycles'][n]
        model.save_interval = options['save_interval']


class FourierSettings(SettingsPanel):
    backend = FourierVisions
    
    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Image scale', 'scale', 1, 16, steps=1, default=8),
            Slider(self, 'Save every', 'save_interval', 20, 1000, steps=20, default=200),
            Slider(self, 'Rough cycles', 'cycles', 100, 5000, steps=100, default=1000),
            Slider(self, 'Fine cycles', 'cycles', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()

    def apply_options(self, model):
        options = self.get_settings()
        model.set_scale(options['scale']/8)
        model.stages[0]['cycles'] = options['cycles'][0]
        model.stages[1]['cycles'] = options['cycles'][1]
        model.save_interval = options['save_interval']


'''
class CLIPCPPNSettings(SettingsPanel):
    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Save every', 'save_interval', 20, 1000, steps=20, default=200),
            Slider(self, 'Rough cycles', 'cycles', 100, 5000, steps=100, default=1000),
            Slider(self, 'Fine cycles', 'cycles', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()
'''
