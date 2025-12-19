import pygame
import numpy as np
from ui_manager import get_font

class DigitalHeatmap:
    def __init__(self, agent, x, y, width, height):
        self.agent = agent
        self.rect = pygame.Rect(x, y, width, height)
        self.font = get_font(10)
        
        # Pre-calculate state mapping if possible, or just use direct index 0-728
        # State is (s1, s2, s3, s4, s5, speed)
        # s1-s5 are 0-2 (3 values)
        # speed is 0-2 (3 values)
        # Total states = 3^6 = 729
        self.num_states = 729
        self.num_actions = len(agent.actions) # Should be 5

    def get_color(self, q_value, min_q, max_q):
        # Normalize q_value to 0-1
        if max_q == min_q:
            norm = 0.5
        else:
            norm = (q_value - min_q) / (max_q - min_q)
        
        # Clamp
        norm = max(0.0, min(1.0, norm))
        
        # Gradient: Dark Blue (0, 0, 50) -> Bright Green (0, 255, 0) -> Gold (255, 215, 0)
        # Let's do a simple 2-stop gradient for "Hot/Cold"
        # Low (0.0): (0, 0, 50) - Dark Blue
        # Mid (0.5): (0, 100, 100) - Teal
        # High (1.0): (0, 255, 0) - Bright Green
        # Very High (Bonus): Gold?
        
        # Let's stick to the request: Dark Blue/Black = Low, Bright Green/Gold = High
        
        if norm < 0.5:
            # Black/Blue to Greenish
            # 0.0 -> (0, 0, 30)
            # 0.5 -> (0, 128, 0)
            r = 0
            g = int(255 * (norm * 2)) # 0 to 255
            b = int(60 * (1 - norm * 2)) # 60 to 0
        else:
            # Green to Gold
            # 0.5 -> (0, 255, 0)
            # 1.0 -> (255, 215, 0)
            factor = (norm - 0.5) * 2
            r = int(255 * factor)
            g = 255 - int(40 * factor) # Slight dip in green for gold
            b = 0
            
        return (r, g, b)

    def draw(self, screen, current_state_idx=None, current_action=None):
        # Draw Background
        pygame.draw.rect(screen, (20, 20, 20), self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        
        # Title
        title = self.font.render("Q-Table Heatmap", True, (200, 200, 200))
        screen.blit(title, (self.rect.x, self.rect.y - 15))

        # We need to visualize 729 rows x 5 columns
        # To fit in the rect, we scale.
        
        # Get all Q-values to find min/max for normalization
        # This might be slow every frame, maybe update periodically or estimate
        # For now, let's scan.
        all_q = list(self.agent.q_table.values())
        if not all_q:
            min_q, max_q = -1, 1
        else:
            min_q = min(all_q)
            max_q = max(all_q)
            # Ensure a range
            if max_q - min_q < 1:
                max_q = min_q + 1

        # Create a surface for the heatmap
        # We'll make it 5 pixels wide and 729 pixels tall, then scale it
        # Actually, 5 columns is very narrow. Let's make it 50x729 (10px per col)
        # But we want to scale it to self.rect
        
        # Let's draw directly to a surface of size (self.num_actions, self.num_states)
        # Then scale to self.rect.size
        
        # Optimization: Only update surface if data changed? 
        # For "Animation" effect of heating up, we want to see it change.
        
        # Since 729 is large, let's try to draw it pixel by pixel on a small surface
        map_surf = pygame.Surface((self.num_actions, self.num_states))
        
        # Lock surface for pixel access
        map_surf.lock()
        
        # We need to iterate over all possible states to fill the grid
        # State encoding: (s1, s2, s3, s4, s5, speed)
        # We can just iterate 0 to 728 if we know the mapping.
        # The agent uses a dict `q_table[(state_tuple, action)] = value`
        # We need to reconstruct the state tuple from index 0-728
        
        for i in range(self.num_states):
            # Reconstruct state tuple from index i
            # This depends on how we define the order. 
            # Let's assume standard base-3 counting:
            # s1*3^5 + s2*3^4 ...
            # But the agent just uses the tuple as key.
            # We need a consistent ordering for the heatmap.
            # Let's generate the tuple:
            temp = i
            speed = temp % 3
            temp //= 3
            s5 = temp % 3
            temp //= 3
            s4 = temp % 3
            temp //= 3
            s3 = temp % 3
            temp //= 3
            s2 = temp % 3
            temp //= 3
            s1 = temp % 3
            
            state_tuple = (s1, s2, s3, s4, s5, speed)
            
            for action in range(self.num_actions):
                q = self.agent.get_q_value(state_tuple, action)
                color = self.get_color(q, min_q, max_q)
                map_surf.set_at((action, i), color)
                
        map_surf.unlock()
        
        # Scale to fit the rect
        scaled_surf = pygame.transform.scale(map_surf, (self.rect.width, self.rect.height))
        screen.blit(scaled_surf, self.rect.topleft)
        
        # Highlight current state
        if current_state_idx is not None:
            # Map state index to y position
            # y = (index / 729) * height
            row_height = self.rect.height / self.num_states
            y_pos = self.rect.y + (current_state_idx / self.num_states) * self.rect.height
            
            # Draw a line or box highlighting this row
            # Since it might be sub-pixel, let's draw a visible line
            pygame.draw.rect(screen, (255, 255, 255), (self.rect.x, y_pos, self.rect.width, max(1, row_height * 5)), 1)
            
            # Highlight chosen action pixel
            if current_action is not None:
                col_width = self.rect.width / self.num_actions
                x_pos = self.rect.x + current_action * col_width
                
                highlight_rect = pygame.Rect(x_pos, y_pos, col_width, max(1, row_height * 5))
                pygame.draw.rect(screen, (255, 0, 255), highlight_rect, 2) # Magenta highlight

    def get_state_index(self, state):
        # Convert state tuple to index 0-728
        # State: (s1, s2, s3, s4, s5, speed)
        # Assuming base-3
        s1, s2, s3, s4, s5, speed = state
        idx = s1 * (3**5) + s2 * (3**4) + s3 * (3**3) + s4 * (3**2) + s5 * 3 + speed
        return idx
