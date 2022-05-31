
import io
import re
import threading
from functools import partial
from os.path import dirname
from queue import Queue
from time import sleep

from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')
from julia_set_image import save_julia_set_image, save_img_to_file
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.logger import Logger
from kivy.properties import (NumericProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

# examples:
# 1 - x**2 + x**2 / (2 + 4 * x) + 0.7885 * np.e**(a * 1j)
# 1 - x + x**2 + 0.7885 * np.e**(a * 1j)
# x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) - 0.5885 * np.e**(a * 1j)
# x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) + 0.755534*math.cos(a) + 0.737292*1j*math.cos(a) - 2*0.737292*1j
# 2**x + 0.2885 * np.e**(a * 1j)
# x**2 + 0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)
# x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767
# x**2 + a*.01 - a*.3*1j
# compatible color maps can be found at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html

curdir = dirname(__file__)


class JuliaGrid(GridLayout):
    curr_dir = dirname(__file__)
    status_box = None
    status_label = None
    curr_img = None

    def set_image_texture_property(self, value, *largs):
        self.ids.image.texture = value.texture
        
    def set_status_text_property(self, value, *largs):
        self.status_label.text = value
        
    def set_curr_img_property(self, value, *largs):
        self.curr_img = value
    
    def do_action(self, *args, **kwargs):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args, **kwargs)
        return super(JuliaGrid, self).add_widget(*args, **kwargs)

    def save_julia_set(self):
        filename = str(self.ids.filename.text)
        expr = str(self.ids.expr.text)
        iterations = self.ids.iterations.input_value
        cmap = str(self.ids.cmap.text)
        real_range_min = self.ids.coordinates.x_val - self.ids.zoom.input_value
        real_range_max = self.ids.coordinates.x_val + self.ids.zoom.input_value
        imag_range_min = self.ids.coordinates.y_val - self.ids.zoom.input_value * (self.ids.aspect_ratio.y_val / self.ids.aspect_ratio.x_val)
        imag_range_max = self.ids.coordinates.y_val + self.ids.zoom.input_value * (self.ids.aspect_ratio.y_val / self.ids.aspect_ratio.x_val)
        image_width = int(self.ids.image_width.input_value)
        image_height = int(self.ids.image_width.input_value * self.ids.aspect_ratio.y_val / self.ids.aspect_ratio.x_val)
        filename = filename.replace('/', '\\')

        if image_width * image_height < 178956970:
            self.curr_img = None
            save_thread = PlotThread(self, 1, 'save_thread', args=(filename, expr, real_range_min, real_range_max, imag_range_min, imag_range_max, image_width, image_height, False, 0, iterations, cmap))
            save_thread.start()
            if self.status_box is None:
                self.status_box = BoxLayout(size_hint_y=None, height='48dp')
                self.ids.option_box.add_widget(self.status_box)
                self.status_label = Label(text='Generating Image')
                self.status_box.add_widget(self.status_label)
            else:
                self.status_label.text = 'Generating Image'
        sleep(.1)

    def save_to_file(self):
        filename = str(self.ids.filename.text)
        img = self.curr_img
        if img is not None:
            save_img_to_file(img, filename)


class AspectRatioInput(TextInput):
    x_val = NumericProperty(16)
    y_val = NumericProperty(9)
    pat = re.compile('[^0-9]')

    def insert_text(self, substring: str, from_undo=False):
        pat = self.pat
        if ':' in self.text:
            s = re.sub(pat, '', substring)
        else:
            s = ':'.join(
                re.sub(pat, '', s)
                for s in substring.split(':', 1)
            )

        ret_val = super().insert_text(s, from_undo=from_undo)
        split = self.text.split(':', 1)
        if split[0] != '':
            self.x_val = int(split[0])
        if split[-1] != '':
            self.y_val = int(split[-1])
        return ret_val

    def keyboard_on_key_up(self, window, keycode):
        split = self.text.split(':', 1)
        if split[0] != '':
            self.x_val = int(split[0])
        if split[-1] != '':
            self.y_val = int(split[-1])
        return super().keyboard_on_key_up(window, keycode)


class CoordinateInput(TextInput):
    x_val = NumericProperty(0.0)
    y_val = NumericProperty(0.0)
    pat = re.compile('[^0-9.-]')

    def insert_text(self, substring: str, from_undo=False):
        pat = self.pat
        if ',' in self.text:
            s = re.sub(pat, '', substring)
        else:
            s = ','.join(
                re.sub(pat, '', s)
                for s in substring.split(',', 1)
            )

        ret_val = super().insert_text(s, from_undo=from_undo)
        split = self.text.split(',', 1)
        if split[0] != '' and split[0] != '.' and len(re.findall(r'[.]', split[0])) < 2:
            self.x_val = float(split[0])
        if split[-1] != '' and split[-1] != '.' and len(re.findall(r'[.]', split[-1])) < 2:
            self.y_val = float(split[-1])
        return ret_val

    def keyboard_on_key_up(self, window, keycode):
        split = self.text.split(',', 1)
        if split[0] != '' and split[0] != '.' and len(re.findall(r'[.]', split[0])) < 2:
            self.x_val = float(split[0])
        if split[-1] != '' and split[-1] != '.' and len(re.findall(r'[.]', split[-1])) < 2:
            self.y_val = float(split[-1])
        return super().keyboard_on_key_up(window, keycode)


class IntInput(TextInput):
    input_value = NumericProperty(1920)

    def insert_text(self, substring: str, from_undo=False):
        ret_val = super().insert_text(substring, from_undo=from_undo)
        if self.text != '':
            self.input_value = int(self.text)
        return ret_val

    def keyboard_on_key_up(self, window, keycode):
        if self.text != '':
            self.input_value = int(self.text)
        return super().keyboard_on_key_up(window, keycode)


class FloatInput(TextInput):
    input_value = NumericProperty(1.0)

    def insert_text(self, substring: str, from_undo=False):
        ret_val = super().insert_text(substring, from_undo=from_undo)
        if self.text != '' and self.text != '.':
            self.input_value = float(self.text)
        return ret_val

    def keyboard_on_key_up(self, window, keycode):
        if self.text != '' and self.text != '.':
            self.input_value = float(self.text)
        return super().keyboard_on_key_up(window, keycode)


class JuliaApp(App):
    def build(self):
        grid = JuliaGrid(rows=1, cols=2)
        return grid


class PlotThread(threading.Thread):
    def __init__(self, grid: JuliaGrid, threadID, name, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.args = args
        self.grid = grid

    def run(self):
        img = save_julia_set_image(*self.args)
        Clock.schedule_once(partial(self.grid.set_curr_img_property, img))
        while(self.grid.curr_img is None):
            ...
        Clock.schedule_once(partial(self.grid.set_status_text_property, 'Done'))
        buffer = io.BytesIO()
        
        self.grid.curr_img.save(buffer, format='png')
        buffer.seek(0)
        im = CoreImage(io.BytesIO(buffer.read()), ext='png')
        Clock.schedule_once(partial(self.grid.set_image_texture_property, im))


if __name__ == '__main__':
    JuliaApp().run()
