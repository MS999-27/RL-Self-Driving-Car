import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from ai import SAC

# Set input configuration
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Global Variables
last_x, last_y, n_points, length = 0, 0, 0, 0
brain = SAC(5) 
last_reward = 0
scores = []
first_update = True
longueur, largeur = 0, 0

def init():
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((longueur, largeur))
    goal_x, goal_y = 20, largeur - 20
    first_update = False

last_distance = 0

class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x, sensor1_y = NumericProperty(0), NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x, sensor2_y = NumericProperty(0), NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x, sensor3_y = NumericProperty(0), NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    
    # FIX: ObjectProperty allows NumPy floats without crashing Kivy
    signal1 = ObjectProperty(0.0)
    signal2 = ObjectProperty(0.0)
    signal3 = ObjectProperty(0.0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = float(rotation)
        self.angle = float(self.angle + self.rotation)
        
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
        for s in ['sensor1', 'sensor2', 'sensor3']:
            sx, sy = getattr(self, s+'_x'), getattr(self, s+'_y')
            if 10 < sx < longueur-10 and 10 < sy < largeur-10:
                # Use .item() to extract a pure Python float from NumPy sum
                raw_sum = np.sum(sand[int(sx)-10:int(sx)+10, int(sy)-10:int(sy)+10])
                # Use .item() to strip the NumPy formatting entirely
    val = float(np.sum(sand[int(sx)-10:int(sx)+10, int(sy)-10:int(sy)+10]).item()) / 400.0
    setattr(self, 'signal'+s[-1], val)
            else:
                setattr(self, 'signal'+s[-1], 1.0)

class Ball1(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(1, 0, 0)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=self.update_ellipse)
    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos

class Ball2(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0, 1, 0)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=self.update_ellipse)
    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos

class Ball3(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0, 0, 1)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=self.update_ellipse)
    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos

class Game(Widget):
    # These placeholders remain, but we will fill them manually
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        # Safety check: if for some reason car is still None, we skip to avoid crash
        if self.car is not None:
            self.car.center = self.center
            self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, longueur, largeur
        longueur, largeur = self.width, self.height
        if first_update:
            init()

        # Ensure the car exists before trying to run AI logic
        if self.car is None:
            return

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [float(self.car.signal1), float(self.car.signal2), float(self.car.signal3), orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        self.car.move(action)
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -10
        else: 
            self.car.velocity = Vector(4, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.5

        if self.car.x < 10 or self.car.x > self.width - 10 or self.car.y < 10 or self.car.y > self.height - 10:
            last_reward = -10
        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        last_distance = distance

class CarApp(App):
    def build(self):
        # 1. Create the Main Game Layout
        parent = Game()
        
        # 2. Create the ACTUAL Car and Sensor objects
        moving_car = Car()
        sensor_red = Ball1()
        sensor_green = Ball2()
        sensor_blue = Ball3()
        
        # 3. ATTACH them to the parent so they appear on screen
        parent.add_widget(moving_car)
        parent.add_widget(sensor_red)
        parent.add_widget(sensor_green)
        parent.add_widget(sensor_blue)
        
        # 4. CRITICAL: Manually link the names to the objects
        # This prevents the 'NoneType' error
        parent.car = moving_car
        parent.ball1 = sensor_red
        parent.ball2 = sensor_green
        parent.ball3 = sensor_blue
        
        # 5. Now it is finally safe to call serve_car
        parent.serve_car()
        
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        parent.add_widget(self.painter)
        return parent

if __name__ == '__main__':
    CarApp().run()
