from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import os

from kivy.utils import get_color_from_hex

from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color
from kivy.graphics import RenderContext
from kivy.uix.behaviors import ButtonBehavior

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

class ResultPopup(Popup):
    def __init__(self,pathfile,color ,Text ='Please wait a moment, while X-Viewer is analysing your image', **kwargs):
        super().__init__(**kwargs)


        # Add the popup to the new window



        self.popuplayout = BoxLayout(orientation='vertical', padding=(50, 50), spacing=0)
        self.ButtonsLayout = BoxLayout(orientation='horizontal', padding=(50, 50), spacing=50)

        self.Image_Placeholder = Image(source=f'{pathfile}', allow_stretch=False, keep_ratio=True)

        self.red_danger = get_color_from_hex('#C70039')
        self.blue = get_color_from_hex('#6495ED')

        if color == "red":

            self.Pneumonia_Text_Placeholder = Label(text=Text,color=self.red_danger)
        else:
            self.Pneumonia_Text_Placeholder = Label(text=Text)

        self.close_Popup_Button = Button(text='Close',size_hint=(0.5, 1), size=(50, 25), background_color=self.blue,

                                         on_press=self.dismiss)

        self.popuplayout.add_widget(self.Image_Placeholder)
        self.popuplayout.add_widget(self.Pneumonia_Text_Placeholder)
        self.ButtonsLayout.add_widget(self.close_Popup_Button)
        self.popuplayout.add_widget(self.ButtonsLayout)
        self.title = "Result"
        self.content = self.popuplayout



class XRayVision(App):
    def build(self):

        #Load the model only once
        self.model =  load_model('./assets/chest_xray.h5')

        #Colors
        self.blue = get_color_from_hex('#6495ED')
        self.red_danger = get_color_from_hex('#C70039')
        self.black = get_color_from_hex("#130006")
        self.grenat = get_color_from_hex("#A91C48")
        self.purple = get_color_from_hex("#962AB6")
        # Title + Two import buttons
        self.bg = FloatLayout()
        bg_image = Image(source='assets/bg.jpg', size_hint=(1, 1))
        self.bg.add_widget(bg_image)

        self.layout = BoxLayout(orientation='vertical', padding=(0, 0), spacing=0)
        self.bg.add_widget(self.layout)
        label = Label(text='PNEUMONIA DETECTION', font_size=20)
        self.ButtonsLayout = BoxLayout(orientation='horizontal', padding=(50, 50), spacing=50)
        self.layout.add_widget(label)
        self.layout.add_widget(self.ButtonsLayout)
        self.ButtonsLayout.add_widget(Button(text='Import Image', size_hint=(0.5, 1), size=(75, 50),
                                             background_color=self.purple,on_press=self.import_image,

                                             font_size=20
                                             )
                                      )
        self.ButtonsLayout.add_widget(Button(text='Take Live Image', size_hint=(0.5, 1), size=(75, 50),
                                             background_color=self.grenat,
                                             font_size=20
                                             )
                                      )

        self.filename_label = Label(text='No file selected', font_size=20,color=self.black)
        self.layout.add_widget(self.filename_label)

        #Buttons layout
        self.Pneumonia_layout = BoxLayout(orientation='horizontal', padding=(50, 50), spacing=50)

        #Popup Layout
        self.Pneumonia_result_Layout = BoxLayout(orientation='vertical', padding=(50, 50), spacing=50)

        self.Pneumonia_button = Button(text='Start Pneumonia Detection', size_hint=(0.5, 1), size=(50, 25),
                                       background_color=self.black ,
                                       font_size=20,
                                       center_x=self.layout.width / 2, center_y=self.layout.height / 2,
                                       on_press=self.detection)



        return self.bg



    def import_image(self, instance):
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        chooser = FileChooserIconView(path=desktop)
        chooser.bind(on_selection=self.select_image)
        chooser.bind(on_submit=self.submit)
        self.popup = Popup(title='Import X-ray Image', content=chooser, size_hint=(0.9, 0.9))
        self.popup.open()


    def select_image(self, instance, selection):
        if len(selection) > 0:
            self.image.source = selection[0]
            print(self.image.source)

    def submit(self, instance, path, selection):
         print(str(path))
         if str(path) !='[]':
             pathname = str(path[0])

             extension = pathname.split(".")[-1]

             self.selected_image_path = str(path)
             self.selected_image_path = self.selected_image_path[2:-2]
             self.filename_label.text = self.selected_image_path


             if self.Pneumonia_button.parent is None:
                     # If not, add it to the Pneumonia_layout parent
                self.Pneumonia_layout.add_widget(self.Pneumonia_button)
                self.layout.add_widget(self.Pneumonia_layout)


         self.popup.dismiss()

    def detection(self,instance):
        self.Pneumonia_button.parent.remove_widget(self.Pneumonia_button)
        self.DetectingPneumonia(self.selected_image_path)

        # Check if the Pneumonia_button widget is already a child of the Pneumonia_layout parent
        if self.Pneumonia_button.parent is None:
            # If not, add it to the Pneumonia_layout parent
            self.Pneumonia_layout.remove_widget(self.Pneumonia_button)
            self.layout.remove_widget(self.Pneumonia_layout)

    def DetectingPneumonia(self,imagepath):
        img = tf.keras.utils.load_img(f'{imagepath}', target_size=(224, 224))
        x=tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = self.model.predict(img_data)
        result = classes[0][0]
        if result > 0.9:
            result_popup = ResultPopup(pathfile=self.selected_image_path,color="white", Text="Result is Normal")
            self.close_popup()
            result_popup.open()

        else:
            result_popup = ResultPopup(pathfile=self.selected_image_path, color="red",Text="Person is Affected by pneumonia")
            self.close_popup()
            result_popup.open()





    def close_popup(self):
        self.filename_label.text ='No file selected';
        self.layout.remove_widget(self.Pneumonia_layout)
        self.Pneumonia_layout.remove_widget(self.Pneumonia_button)



if __name__ == '__main__':
    XRayVision().run()