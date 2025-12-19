import pygame

class Map:
    def __init__(self, name, blocks, finish_line):
        self.name = name
        self.blocks = blocks
        self.finish_line = finish_line

# Constants for track dimensions (must match game.py)
TRACK_WIDTH = 800
TRACK_HEIGHT = 600

# 1. Simple Circuit (The original one)
SIMPLE_CIRCUIT = Map(
    "Simple Circuit",
    [
        pygame.Rect(100, 0, 20, TRACK_HEIGHT),   
        pygame.Rect(680, 0, 20, TRACK_HEIGHT),   
    ],
    pygame.Rect(120, 50, 560, 20)
)

# 2. Reverse Circuit (Same blocks, finish line at bottom)
REVERSE_CIRCUIT = Map(
    "Reverse Circuit",
    [
        pygame.Rect(100, 0, 20, TRACK_HEIGHT),   
        pygame.Rect(680, 0, 20, TRACK_HEIGHT),   
    ],
    pygame.Rect(120, 550, 560, 20)
)

AVAILABLE_MAPS = [SIMPLE_CIRCUIT, REVERSE_CIRCUIT]
