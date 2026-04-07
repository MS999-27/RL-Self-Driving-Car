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
last_distance = 0

def init():
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((longueur, largeur))
    goal_x, goal_y = 20, largeur - 20
    first_update = False

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
                raw_sum = np.sum(sand[int(sx)-10:int(sx)+10, int(sy)-10:int(sy)+10])
                val = float(raw_sum.item()) / 400.0
                setattr(self, 'signal' + s[-1], val)
            else:
                setattr(self, 'signal' + s[-1], 1.0)

class Ball1(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(1, 0, 0)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=lambda obj, pos: setattr(self.ellipse, 'pos', pos))

class Ball2(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0, 1, 0)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=lambda obj, pos: setattr(self.ellipse, 'pos', pos))

class Ball3(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0, 0, 1)
            self.ellipse = Ellipse(pos=self.pos, size=(10, 10))
        self.bind(pos=lambda obj, pos: setattr(self.ellipse, 'pos', pos))

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x, last_y = int(touch.x), int(touch.y)
            sand[last_x, last_y] = 1

    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x, y = int(touch.x), int(touch.y)
            sand[int(x)-10:int(x)+10, int(y)-10:int(y)+10] = 1
            last_x, last_y = x, y

class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        # SAFETY GATE: Only runs if car is actually connected
        if self.car:
            self.car.center = self.center
            self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, longueur, largeur
        longueur, largeur = self.width, self.height
        if first_update:
            init()

        if not self.car: return # Prevent crash

        xx, yy = goal_x - self.car.x, goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [float(self.car.signal1), float(self.car.signal2), float(self.car.signal3), orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        self.car.move(action)
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos, self.ball2.pos, self.ball3.pos = self.car.sensor1, self.car.sensor2, self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -10
        else: 
            self.car.velocity = Vector(4, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance: last_reward = 0.5

        if self.car.x < 10 or self.car.x > self.width - 10 or self.car.y < 10 or self.car.y > self.height - 10:
            last_reward = -10
        if distance < 100:
            goal_x, goal_y = self.width - goal_x, self.height - goal_y
        last_distance = distance

class SelfDrivingApp(App): # Renamed to break the link to car.kv
    def build(self):
        parent = Game()
        self.car_obj = Car()
        self.b1, self.b2, self.b3 = Ball1(), Ball2(), Ball3()
        
        parent.add_widget(self.car_obj)
        parent.add_widget(self.b1)
        parent.add_widget(self.b2)
        parent.add_widget(self.b3)
        
        # Explicit link
        parent.car = self.car_obj
        parent.ball1, parent.ball2, parent.ball3 = self.b1, self.b2, self.b3
        
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        parent.add_widget(MyPaintWidget())
        return parent

if __name__ == '__main__':
    SelfDrivingApp().run()
