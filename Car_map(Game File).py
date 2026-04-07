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

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Global Variables
last_x, last_y, n_points, length = 0, 0, 0, 0
brain = SAC(5) # 5 Sensors/Inputs
last_reward = 0
scores = []
first_update = True

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
    signal1, signal2, signal3 = NumericProperty(0), NumericProperty(0), NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
        # Density sensing
        for s in ['sensor1', 'sensor2', 'sensor3']:
            sx, sy = getattr(self, s+'_x'), getattr(self, s+'_y')
            if 10 < sx < longueur-10 and 10 < sy < largeur-10:
                val = np.sum(sand[int(sx)-10:int(sx)+10, int(sy)-10:int(sy)+10])/400.
                setattr(self, 'signal'+s[-1], val)
            else:
                setattr(self, 'signal'+s[-1], 1.0)

class Ball1(Widget): pass
class Ball2(Widget): pass
class Ball3(Widget): pass

class Game(Widget):
    car = ObjectProperty(None)
    ball1, ball2, ball3 = ObjectProperty(None), ObjectProperty(None), ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, longueur, largeur
        longueur, largeur = self.width, self.height
        if first_update: init()

        xx, yy = goal_x - self.car.x, goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        
        # SAC Output (Smooth Rotation)
        rotation = brain.update(last_reward, last_signal)
        self.car.move(rotation)

        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos, self.ball2.pos, self.ball3.pos = self.car.sensor1, self.car.sensor2, self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -10
        else:
            self.car.velocity = Vector(4, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance: last_reward = 0.5

        if self.car.x < 10 or self.car.x > self.width-10 or self.car.y < 10 or self.car.y > self.height-10:
            last_reward = -10
        
        if distance < 100:
            goal_x, goal_y = self.width-goal_x, self.height-goal_y
        last_distance = distance

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 20)
            last_x, last_y = int(touch.x), int(touch.y)
            sand[last_x, last_y] = 1

    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            sand[int(touch.x)-10:int(touch.x)+10, int(touch.y)-10:int(touch.y)+10] = 1

class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        parent.add_widget(self.painter)
        return parent

if __name__ == '__main__':
    CarApp().run()



        
       
            
     
      
