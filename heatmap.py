import pygame
import numpy as np
from ui_manager import get_font

class SensorActionHeatmap:
    def __init__(self, agent, x, y, width, height):
        self.agent = agent
        self.rect = pygame.Rect(x, y, width, height)
        self.font = get_font(12)
        self.msg_font = get_font(10)
        
        # Grid definition
        # Rows: Front Sensor Distance (0: Close, 1: Medium, 2: Far)
        # Cols: Steering Actions (3: Left, 0: Straight, 4: Right) - subset of actions
        self.rows = [0, 1, 2] 
        self.row_labels = ["Close <40", "Med <100", "Far >100"]
        
        self.cols = [3, 0, 4] # Action indices
        self.col_labels = ["Left", "Straight", "Right"]
        
    def get_color(self, q_value, min_q, max_q):
        # Normalize q_value to 0-1
        if max_q == min_q:
            norm = 0.5
        else:
            norm = (q_value - min_q) / (max_q - min_q)
        
        norm = max(0.0, min(1.0, norm))
        
        # Red (Low) to Green (High)
        # 0.0 -> (255, 0, 0)
        # 0.5 -> (255, 255, 0)
        # 1.0 -> (0, 255, 0)
        
        if norm < 0.5:
            # Red to Yellow
            # 0.0 -> (255, 0, 0)
            # 0.5 -> (255, 255, 0)
            n = norm * 2
            r = 255
            g = int(255 * n)
            b = 0
        else:
            # Yellow to Green
            # 0.5 -> (255, 255, 0)
            # 1.0 -> (0, 255, 0)
            n = (norm - 0.5) * 2
            r = int(255 * (1 - n))
            g = 255
            b = 0
            
        return (r, g, b)

    def draw(self, screen, current_state=None, current_action=None):
        # Draw Background
        pygame.draw.rect(screen, (30, 30, 30), self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 1)
        
        if current_state is None:
            msg = self.msg_font.render("Waiting for state...", True, (200, 200, 200))
            screen.blit(msg, (self.rect.centerx - msg.get_width()//2, self.rect.centery))
            return

        # Layout calculations
        # Need space for labels
        margin_left = 60
        margin_bottom = 30
        grid_width = self.rect.width - margin_left
        grid_height = self.rect.height - margin_bottom
        
        cell_w = grid_width / len(self.cols)
        cell_h = grid_height / len(self.rows)
        
        # Find min/max Q for THIS slice
        # To make colors meaningful across the grid
        q_values = []
        
        # We need to construct hypothetical states
        # current_state = (s1, s2, s3, s4, s5, speed)
        # s3 is Front Sensor (index 2)
        
        for r_idx, dist_val in enumerate(self.rows):
            for c_idx, action_val in enumerate(self.cols):
                # Construct synthetic state
                # Replace index 2 with dist_val
                syn_state_list = list(current_state)
                syn_state_list[2] = dist_val 
                syn_state = tuple(syn_state_list)
                
                q = self.agent.get_q_value(syn_state, action_val)
                q_values.append(q)
                
        if not q_values:
            min_q, max_q = -1, 1
        else:
            min_q = min(q_values)
            max_q = max(q_values)
            if max_q == min_q: max_q += 1

        # Draw Grid
        for r_idx, dist_val in enumerate(self.rows):
            for c_idx, action_val in enumerate(self.cols):
                # Construct state
                syn_state_list = list(current_state)
                syn_state_list[2] = dist_val
                syn_state = tuple(syn_state_list)
                
                q = self.agent.get_q_value(syn_state, action_val)
                color = self.get_color(q, min_q, max_q)
                
                # Logic: Row 0 is Close (Top or Bottom?)
                # Usually graphs have 0 at bottom. So let's reverse row index for Y.
                # Or user asked: "Short Distance" rows turn red.
                # Let's put "Far" at Top, "Close" at Bottom? 
                # Or "Close" at Top (Danger)?
                # Let's follow standard matrix: Row 0 at top.
                # Row 0 = Close (dist 0).
                
                x = self.rect.x + margin_left + c_idx * cell_w
                y = self.rect.y + r_idx * cell_h
                
                pygame.draw.rect(screen, color, (x, y, cell_w, cell_h))
                pygame.draw.rect(screen, (0,0,0), (x, y, cell_w, cell_h), 1)
                
                # Draw Q value text
                q_txt = self.msg_font.render(f"{q:.1f}", True, (0, 0, 0))
                screen.blit(q_txt, (x + cell_w/2 - q_txt.get_width()/2, y + cell_h/2 - q_txt.get_height()/2))
                
                # Highlight if this matches CURRENT real outcome
                if current_state[2] == dist_val and current_action == action_val:
                     pygame.draw.rect(screen, (255, 255, 255), (x+2, y+2, cell_w-4, cell_h-4), 3)

        # Draw Labels
        # Y Axis (Rows)
        for r_idx, label in enumerate(self.rows):
             txt = self.font.render(self.row_labels[r_idx], True, (200, 200, 200))
             # Center vertically in row
             y = self.rect.y + r_idx * cell_h + cell_h/2 - txt.get_height()/2
             screen.blit(txt, (self.rect.x + 5, y))
             
        # X Axis (Cols)
        for c_idx, label in enumerate(self.cols):
            txt = self.font.render(self.col_labels[c_idx], True, (200, 200, 200))
            # Center horizontally
            x = self.rect.x + margin_left + c_idx * cell_w + cell_w/2 - txt.get_width()/2
            screen.blit(txt, (x, self.rect.bottom - 20))

        # Helper Text
        speed_val = current_state[5] # Speed index
        speed_lbl = ["Stop/Slow", "Med", "Fast"][speed_val]
        curr_txt = self.font.render(f"Current Speed: {speed_lbl}", True, (255, 255, 255))
        screen.blit(curr_txt, (self.rect.x, self.rect.y - 20))
