from tkinter import Frame

from models.visions import PyramidVisions, DirectVisions, FourierVisions
from models.lucidrains import DeepDaze, BigSleep

from gui.gui_elements import Slider, ExtraPrompts


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
    group = 'Visions'

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

    def get_stages(self):
        options = self.get_settings()
        return options['cycles']


class DirectSettings(SettingsPanel):
    backend = DirectVisions
    group = 'Visions'

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

    def get_stages(self):
        options = self.get_settings()
        return options['cycles']


class FourierSettings(SettingsPanel):
    backend = FourierVisions
    group = 'Visions'
    
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

    def get_stages(self):
        options = self.get_settings()
        return options['cycles']


class DazeSettings(SettingsPanel):
    backend = DeepDaze
    group = 'Lucidrains'
    
    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Save every', 'save_interval', 10, 100, steps=10, default=20),
            Slider(self, 'Image size', 'scale', 128, 1024, steps=32, default=256),
            Slider(self, 'Layers', 'layers', 8, 32, steps=1, default=16),
            Slider(self, 'Epochs', 'epochs', 1, 10, steps=1, default=5),
            Slider(self, 'Iterations', 'iterations', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()

    def apply_options(self, model):
        options = self.get_settings()
        model.epochs = options['epochs']
        model.iterations = options['iterations']
        model.save_interval = options['save_interval']
        model.image_size = options['scale']
        model.layers = options['layers']

    def get_stages(self):
        options = self.get_settings()
        return [ options['iterations'] ] * options['epochs']


class SleepSettings(SettingsPanel):
    backend = BigSleep
    group = 'Lucidrains'
    
    def __init__(self, frame):
        SettingsPanel.__init__(self, frame)
        self.options = [
            Slider(self, 'Save every', 'save_interval', 10, 100, steps=10, default=20),
            Slider(self, 'Image size', 'scale', 128, 1024, steps=32, default=256),
            Slider(self, 'Epochs', 'epochs', 1, 10, steps=1, default=5),
            Slider(self, 'Iterations', 'iterations', 100, 5000, steps=100, default=1000)
        ]
        self.pack_options()

    def apply_options(self, model):
        options = self.get_settings()
        model.epochs = options['epochs']
        model.iterations = options['iterations']
        model.save_interval = options['save_interval']
        model.image_size = options['scale']

    def get_stages(self):
        options = self.get_settings()
        return [ options['iterations'] ] * options['epochs']

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
