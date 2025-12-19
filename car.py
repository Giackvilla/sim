import math
import pygame

class Car:
    def __init__(self, x, y, angle=0, length=40, width=20, max_speed=10, acceleration=0.5, steering=5):
        self.x = x
        self.y = y
        self.angle = angle  # Degrees
        self.speed = 0
        self.length = length
        self.width = width
        
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.steering = steering
        self.friction = 0.02 # Reduced friction
        self.alive = True
        self.total_reward = 0

    def accelerate(self):
        self.speed += self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed

    def brake(self):
        self.speed -= self.acceleration
        if self.speed < -self.max_speed/2: # Reverse is slower
            self.speed = -self.max_speed/2

    def steer_left(self):
        self.angle -= self.steering
        self.angle %= 360

    def steer_right(self):
        self.angle += self.steering
        self.angle %= 360

    def update(self):
        # Apply friction
        if self.speed > 0:
            self.speed -= self.friction
            if self.speed < 0: self.speed = 0
        elif self.speed < 0:
            self.speed += self.friction
            if self.speed > 0: self.speed = 0

        # Move
        # Angle 0 is usually Right in math, but Up in games often. 
        # Let's assume 0 is UP (North).
        # sin/cos take radians.
        # If 0 is Up:
        # dx = sin(angle) * speed
        # dy = -cos(angle) * speed (because y decreases going up)
        
        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.speed
        self.y -= math.cos(rad) * self.speed

    def get_sensors(self, blocks, screen_width, screen_height):
        # 5 Rays: -90, -45, 0, 45, 90 degrees relative to heading
        angles = [-90, -45, 0, 45, 90]
        readings = []
        
        for a in angles:
            ray_angle = math.radians(self.angle + a)
            
            # Raycasting
            dist = 0
            max_dist = 200 # Max vision range
            step = 5
            
            found = False
            curr_x, curr_y = self.x, self.y
            
            while dist < max_dist:
                curr_x += math.sin(ray_angle) * step
                curr_y -= math.cos(ray_angle) * step
                dist += step
                
                # Check boundaries (screen edges are walls too if we treat them as such, 
                # but our blocks define the track now. Let's check blocks AND screen edges)
                if curr_x < 0 or curr_x > screen_width or curr_y < 0 or curr_y > screen_height:
                    found = True
                    break
                
                # Check blocks
                point_rect = pygame.Rect(curr_x, curr_y, 1, 1)
                for block in blocks:
                    if point_rect.colliderect(block):
                        found = True
                        break
                
                if found:
                    break
            
            # Discretize distance for simpler state space
            # 0: Close (< 40), 1: Medium (< 100), 2: Far
            if dist < 40:
                readings.append(0)
            elif dist < 100:
                readings.append(1)
            else:
                readings.append(2)
                
        return tuple(readings)
