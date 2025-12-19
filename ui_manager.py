import pygame

# Gemini Dark Theme Colors
COLORS = {
    "BACKGROUND": (19, 19, 20),       # #131314
    "PANEL_BG": (30, 31, 32),         # #1E1F20
    "ACCENT_BLUE": (138, 180, 248),   # #8AB4F8
    "ACCENT_PURPLE": (197, 138, 249), # #C58AF9
    "TEXT_PRIMARY": (227, 227, 227),  # #E3E3E3
    "TEXT_SECONDARY": (154, 160, 166),# #9AA0A6
    "BORDER": (60, 64, 67),           # #3C4043
    "SUCCESS": (129, 201, 149),       # #81C995
    "DANGER": (242, 139, 130),        # #F28B82
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0)
}

import os
FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
ROBOTO_REGULAR = os.path.join(FONT_DIR, "Roboto-Regular.ttf")
ROBOTO_BOLD = os.path.join(FONT_DIR, "Roboto-Bold.ttf")

def get_font(size, bold=False):
    path = ROBOTO_BOLD if bold else ROBOTO_REGULAR
    try:
        return pygame.font.Font(path, size)
    except FileNotFoundError:
        print(f"Warning: Font not found at {path}, falling back to Arial")
        return pygame.font.SysFont("arial", size, bold=bold)

class UIButton:
    def __init__(self, x, y, width, height, text, callback, color=COLORS["PANEL_BG"], hover_color=COLORS["BORDER"], text_color=COLORS["TEXT_PRIMARY"]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = get_font(16, bold=True)
        self.is_hovered = False
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        
        # Draw rounded rect
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, COLORS["BORDER"], self.rect, 1, border_radius=12) # Border
        
        # Text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered and event.button == 1:
                if self.callback:
                    self.callback()
                return True
        return False

class UISlider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label="Slider", is_integer=False, precision=2):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.is_integer = is_integer
        self.precision = precision
        self.dragging = False
        self.font = get_font(14)
        
        # Handle
        self.handle_radius = 10
        self.update_handle_pos()

    def update_handle_pos(self):
        # Map value to x position
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.rect.x + ratio * self.rect.width
        self.handle_y = self.rect.centery

    def update_value_from_pos(self, x):
        # Clamp x
        x = max(self.rect.x, min(self.rect.right, x))
        ratio = (x - self.rect.x) / self.rect.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        
        if self.is_integer:
            self.value = round(self.value)
            
        self.update_handle_pos()

    def draw(self, screen):
        # Label
        if self.is_integer:
            val_str = f"{int(self.value)}"
        else:
            val_str = f"{self.value:.{self.precision}f}"
        label_surf = self.font.render(f"{self.label}: {val_str}", True, COLORS["TEXT_SECONDARY"])
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))
        
        # Track
        pygame.draw.rect(screen, COLORS["BORDER"], (self.rect.x, self.rect.centery - 2, self.rect.width, 4), border_radius=2)
        
        # Filled Track (Left of handle)
        fill_width = self.handle_x - self.rect.x
        pygame.draw.rect(screen, COLORS["ACCENT_BLUE"], (self.rect.x, self.rect.centery - 2, fill_width, 4), border_radius=2)
        
        # Handle
        pygame.draw.circle(screen, COLORS["ACCENT_BLUE"], (int(self.handle_x), int(self.handle_y)), self.handle_radius)
        pygame.draw.circle(screen, COLORS["WHITE"], (int(self.handle_x), int(self.handle_y)), self.handle_radius, 1)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check collision with handle or track
            # Expand hit area slightly
            hit_rect = self.rect.inflate(10, 20)
            if hit_rect.collidepoint(event.pos):
                self.dragging = True
                self.update_value_from_pos(event.pos[0])
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_value_from_pos(event.pos[0])
                return True
        return False

class UIToggle:
    def __init__(self, x, y, label, initial_state=False):
        self.x = x
        self.y = y
        self.label = label
        self.checked = initial_state
        self.font = get_font(14)
        self.rect = pygame.Rect(x, y, 20, 20)
        
    def draw(self, screen):
        # Box
        color = COLORS["ACCENT_BLUE"] if self.checked else COLORS["BORDER"]
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        if self.checked:
            # Checkmark (simple line)
            pygame.draw.line(screen, COLORS["BACKGROUND"], (self.x + 4, self.y + 10), (self.x + 8, self.y + 16), 2)
            pygame.draw.line(screen, COLORS["BACKGROUND"], (self.x + 8, self.y + 16), (self.x + 16, self.y + 4), 2)
            
        # Label
        label_surf = self.font.render(self.label, True, COLORS["TEXT_SECONDARY"])
        screen.blit(label_surf, (self.x + 30, self.y + 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False
