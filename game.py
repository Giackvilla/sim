import pygame
import math
import numpy as np
from car import Car
from agent import QLearningAgent
from agent2 import DQNAgent
from agent2 import DQNAgent
from ui_manager import COLORS, UIButton, UISlider, UIToggle, get_font
from maps import AVAILABLE_MAPS

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 60 

# Layout
SIDEBAR_WIDTH = 250
ANALYSIS_WIDTH = 350
VIEWPORT_WIDTH = SCREEN_WIDTH - SIDEBAR_WIDTH - ANALYSIS_WIDTH
VIEWPORT_HEIGHT = SCREEN_HEIGHT - 40 # Padding
VIEWPORT_X = SIDEBAR_WIDTH + 20
VIEWPORT_Y = 20

# Sim World to Viewport Mapping
# The track is 800x600. Viewport is ~760x760.
# We'll center the 800x600 track in the viewport.
TRACK_WIDTH = 800
TRACK_HEIGHT = 600

def draw_map_preview(screen, rect, map_obj):
    # Draw Background
    pygame.draw.rect(screen, COLORS["BLACK"], rect, border_radius=8)
    pygame.draw.rect(screen, COLORS["BORDER"], rect, 1, border_radius=8)
    
    # Scale Factors
    scale_x = rect.width / TRACK_WIDTH
    scale_y = rect.height / TRACK_HEIGHT
    
    # Draw Blocks
    for block in map_obj.blocks:
        scaled_rect = pygame.Rect(
            rect.x + block.x * scale_x,
            rect.y + block.y * scale_y,
            block.width * scale_x,
            block.height * scale_y
        )
        pygame.draw.rect(screen, COLORS["BORDER"], scaled_rect)
        
    # Draw Finish Line
    f = map_obj.finish_line
    scaled_finish = pygame.Rect(
        rect.x + f.x * scale_x,
        rect.y + f.y * scale_y,
        f.width * scale_x,
        f.height * scale_y
    )
    pygame.draw.rect(screen, COLORS["SUCCESS"], scaled_finish)

def show_menu(screen):
    # Background
    screen.fill(COLORS["BACKGROUND"])
    
    # Title
    font_title = get_font(48, bold=True)
    title = font_title.render("AI Car Simulation", True, COLORS["TEXT_PRIMARY"])
    screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
    
    # State
    selected_agent = None
    selected_map_idx = 0
    num_cars = 10
    
    # UI Elements
    buttons = []
    
    def set_classic():
        nonlocal selected_agent
        selected_agent = "classic"
        
    def set_dqn():
        nonlocal selected_agent
        selected_agent = "dqn"
        
    btn_classic = UIButton(SCREEN_WIDTH//2 - 220, 300, 200, 60, "Classic Q-Learning", set_classic)
    btn_dqn = UIButton(SCREEN_WIDTH//2 + 20, 300, 200, 60, "Deep Q-Network", set_dqn)
    
    slider_cars = UISlider(SCREEN_WIDTH//2 - 150, 450, 300, 20, 1, 100, 10, "Number of Cars", is_integer=True)
    
    # Map Selection UI
    def prev_map():
        nonlocal selected_map_idx
        selected_map_idx = (selected_map_idx - 1) % len(AVAILABLE_MAPS)
        
    def next_map():
        nonlocal selected_map_idx
        selected_map_idx = (selected_map_idx + 1) % len(AVAILABLE_MAPS)
        
    # Arrows
    btn_prev = UIButton(SCREEN_WIDTH//2 - 200, 500, 40, 40, "<", prev_map)
    btn_next = UIButton(SCREEN_WIDTH//2 + 160, 500, 40, 40, ">", next_map)
    
    # Preview Rect
    preview_rect = pygame.Rect(SCREEN_WIDTH//2 - 150, 480, 300, 225) # 4:3 aspect ratio roughly
    # Actually 800x600 is 4:3. 300 width -> 225 height.
    # But we need to fit it. Let's make it smaller.
    preview_rect = pygame.Rect(SCREEN_WIDTH//2 - 120, 500, 240, 180)
    
    # Adjust Start Button Position
    start_clicked = False
    def start_game():
        nonlocal start_clicked
        if selected_agent:
            start_clicked = True
            
    btn_start = UIButton(SCREEN_WIDTH//2 - 100, 720, 200, 50, "START SIMULATION", start_game, color=COLORS["ACCENT_BLUE"], text_color=COLORS["BLACK"])
    
    buttons = [btn_classic, btn_dqn, btn_prev, btn_next, btn_start]
    
    while not start_clicked:
        screen.fill(COLORS["BACKGROUND"])
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
        
        # Update selection visuals
        if selected_agent == "classic":
            btn_classic.color = COLORS["ACCENT_PURPLE"]
            btn_dqn.color = COLORS["PANEL_BG"]
        elif selected_agent == "dqn":
            btn_dqn.color = COLORS["ACCENT_PURPLE"]
            btn_classic.color = COLORS["PANEL_BG"]
            
        for btn in buttons:
            btn.draw(screen)
            
        # Draw Map Preview
        draw_map_preview(screen, preview_rect, AVAILABLE_MAPS[selected_map_idx])
        
        # Draw Map Name
        font_map = get_font(16, bold=True)
        map_name = font_map.render(AVAILABLE_MAPS[selected_map_idx].name, True, COLORS["TEXT_PRIMARY"])
        screen.blit(map_name, (SCREEN_WIDTH//2 - map_name.get_width()//2, preview_rect.bottom + 10))
            
        slider_cars.draw(screen)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, None
            
            for btn in buttons:
                btn.handle_event(event)
            slider_cars.handle_event(event)
            
    return selected_agent, int(slider_cars.value), AVAILABLE_MAPS[selected_map_idx]

def draw_car(screen, car, offset_x, offset_y, is_best=False):
    alpha = 255 if is_best else 50 
    
    car_surface = pygame.Surface((car.width, car.length), pygame.SRCALPHA)
    color = (*COLORS["DANGER"], alpha) # Red-ish
    car_surface.fill(color)
    
    pygame.draw.rect(car_surface, (*COLORS["BLACK"], alpha), (2, 2, car.width-4, 10))
    
    rotated_surface = pygame.transform.rotate(car_surface, -car.angle)
    rect = rotated_surface.get_rect(center=(car.x + offset_x, car.y + offset_y))
    screen.blit(rotated_surface, rect.topleft)

def get_state(car, blocks):
    sensors = car.get_sensors(blocks, TRACK_WIDTH, TRACK_HEIGHT)
    
    if car.speed < 1:
        speed_state = 0
    elif car.speed < 3:
        speed_state = 1
    else:
        speed_state = 2
        
    return sensors + (speed_state,)

def draw_graph(screen, rect, reward_history, completion_history):
    # Draw Panel
    pygame.draw.rect(screen, COLORS["PANEL_BG"], rect, border_radius=12)
    pygame.draw.rect(screen, COLORS["BORDER"], rect, 1, border_radius=12)
    
    # Legend
    font_legend = get_font(12)
    
    # Blue Label
    pygame.draw.rect(screen, COLORS["ACCENT_BLUE"], (rect.x + 150, rect.y + 15, 10, 10))
    lbl_rew = font_legend.render("Reward", True, COLORS["TEXT_SECONDARY"])
    screen.blit(lbl_rew, (rect.x + 165, rect.y + 12))
    
    # Purple Label
    pygame.draw.rect(screen, COLORS["ACCENT_PURPLE"], (rect.x + 220, rect.y + 15, 10, 10))
    lbl_comp = font_legend.render("Completion", True, COLORS["TEXT_SECONDARY"])
    screen.blit(lbl_comp, (rect.x + 235, rect.y + 12))
    
    if not reward_history: return
    
    # Plot Area
    plot_rect = pygame.Rect(rect.x + 10, rect.y + 40, rect.width - 20, rect.height - 50)
    # pygame.draw.rect(screen, COLORS["BACKGROUND"], plot_rect)
    
    # Plot Reward (Blue)
    max_rew = max(max(reward_history), 1)
    min_rew = min(min(reward_history), 0)
    range_rew = max_rew - min_rew if max_rew != min_rew else 1
    
    points_rew = []
    for i, val in enumerate(reward_history):
        x = plot_rect.left + (i / max(1, len(reward_history)-1)) * plot_rect.width
        norm_val = (val - min_rew) / range_rew
        y = plot_rect.bottom - norm_val * plot_rect.height
        points_rew.append((x, y))
        
    if len(points_rew) > 1:
        pygame.draw.lines(screen, COLORS["ACCENT_BLUE"], False, points_rew, 2)
        
    # Plot Completion (Red/Purple)
    if not completion_history: return
    
    points_comp = []
    for i, val in enumerate(completion_history):
        x = plot_rect.left + (i / max(1, len(completion_history)-1)) * plot_rect.width
        y = plot_rect.bottom - (val / 100) * plot_rect.height
        points_comp.append((x, y))
        
    if len(points_comp) > 1:
        pygame.draw.lines(screen, COLORS["ACCENT_PURPLE"], False, points_comp, 2)

def draw_network_or_heatmap(screen, rect, agent, agent_name, heatmap=None, current_state=None, current_action=None):
    # Draw Panel
    pygame.draw.rect(screen, COLORS["PANEL_BG"], rect, border_radius=12)
    pygame.draw.rect(screen, COLORS["BORDER"], rect, 1, border_radius=12)
    
    # Title
    font = get_font(14, bold=True)
    title = font.render("Brain Visualization", True, COLORS["TEXT_SECONDARY"])
    screen.blit(title, (rect.x + 15, rect.y + 15))
    
    content_rect = pygame.Rect(rect.x + 10, rect.y + 40, rect.width - 20, rect.height - 50)
    
    if agent_name == "Q-Table":
        if heatmap:
            # Update heatmap rect to fit content_rect
            heatmap.rect = content_rect
            heatmap.draw(screen, current_state, current_action)
            
    elif agent_name == "DQN (NN)":
        if not hasattr(agent, 'last_activations'): return
        acts = agent.last_activations
        
        # Draw NN with Links
        hidden_show = 8
        layers = [acts['input'], acts['hidden1'][:hidden_show], acts['hidden2'][:hidden_show], acts['output']]
        
        # Calculate positions
        node_positions = []
        for i, layer in enumerate(layers):
            layer_x = content_rect.x + (i / (len(layers)-1)) * content_rect.width
            layer_nodes = []
            for j, val in enumerate(layer):
                layer_y = content_rect.y + (j / max(1, len(layer)-1)) * content_rect.height
                layer_nodes.append((layer_x, layer_y, val))
            node_positions.append(layer_nodes)
            
        # Draw Weights
        weights = [agent.W1, agent.W2, agent.W3]
        for i in range(len(node_positions)-1):
            curr_nodes = node_positions[i]
            next_nodes = node_positions[i+1]
            W = weights[i]
            
            for j, n1 in enumerate(curr_nodes):
                for k, n2 in enumerate(next_nodes):
                    # If hidden layer is truncated, we need to map indices correctly? 
                    # Actually, if we truncate visualization, we should only draw weights for visible nodes.
                    # Since we sliced acts['hidden1'][:8], we are only showing first 8 nodes.
                    # W1 is (state_size, hidden_size). 
                    # If i=0 (Input->H1), j is input idx, k is H1 idx.
                    # We only iterate k up to len(next_nodes) which is 8. So it works.
                    
                    w_val = W[j, k]
                    if math.isnan(w_val): continue
                    
                    if w_val > 0: color = COLORS["SUCCESS"]
                    else: color = COLORS["DANGER"]
                    
                    thickness = 1 + int(abs(w_val) * 2)
                    pygame.draw.line(screen, color, (n1[0], n1[1]), (n2[0], n2[1]), thickness)
        
        # Draw Nodes
        for i, layer in enumerate(node_positions):
            for j, (x, y, val) in enumerate(layer):
                c_val = max(-1, min(1, val))
                if c_val > 0: color = COLORS["SUCCESS"]
                else: color = COLORS["DANGER"]
                
                pygame.draw.circle(screen, color, (int(x), int(y)), 4)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Car Simulation - Gemini UI")
    clock = pygame.time.Clock()
    
    while True: # Main App Loop (Menu -> Game -> Menu)
        agent_choice, num_cars, selected_map = show_menu(screen)
        if agent_choice is None: break
        
        # Game Setup
        blocks = selected_map.blocks
        finish_line = selected_map.finish_line
        actions = [0, 1, 2, 3, 4]
        
        if agent_choice == "classic":
            agent = QLearningAgent(actions)
            agent_name = "Q-Table"
            from heatmap import SensorActionHeatmap
            heatmap = SensorActionHeatmap(agent, 0, 0, 100, 100) # Rect updated later
        else:
            agent = DQNAgent(actions)
            agent_name = "DQN (NN)"
            try: agent.load()
            except: pass
            heatmap = None
            
        episodes = 1000
        NUM_CARS = num_cars
        avg_reward_history = []
        completion_history = []
        
        # UI Components for Game Loop
        should_reset = False
        should_back = False
        
        def on_reset():
            nonlocal should_reset
            should_reset = True
            
        def on_back():
            nonlocal should_back
            should_back = True
            
        btn_reset = UIButton(20, SCREEN_HEIGHT - 140, SIDEBAR_WIDTH - 40, 50, "Reset Training", on_reset, color=COLORS["PANEL_BG"], text_color=COLORS["DANGER"])
        btn_back = UIButton(20, SCREEN_HEIGHT - 70, SIDEBAR_WIDTH - 40, 50, "Back to Menu", on_back)
        
        slider_speed = UISlider(20, 140, SIDEBAR_WIDTH - 40, 20, 1, 50, 1, "Sim Speed (FPS Multiplier)", is_integer=True)
        
        toggle_sensors = UIToggle(20, 190, "Show Sensors", True)
        toggle_brain = UIToggle(20, 230, "Show Brain", True)
        toggle_fast = UIToggle(20, 270, "Fast Mode (No Render)", False)
        
        ui_elements = [btn_reset, btn_back, slider_speed, toggle_sensors, toggle_brain, toggle_fast]
        
        # Episode Loop
        for episode in range(episodes):
            if should_back: break
            if should_reset:
                # Reset agent and history
                if agent_choice == "classic": agent = QLearningAgent(actions)
                else: agent = DQNAgent(actions)
                avg_reward_history = []
                completion_history = []
                should_reset = False
                # Restart episode loop effectively by breaking inner and handling logic? 
                # Easier to just break and let the outer loop restart? 
                # Actually, let's just continue to next episode with reset agent.
            
            cars = [Car(400, TRACK_HEIGHT - 100, angle=0) for _ in range(NUM_CARS)]
            states = [get_state(c, blocks) for c in cars]
            
            done = False
            frame_count = 0
            finished_count = 0
            start_time = pygame.time.get_ticks()
            
            while not done:
                # Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save()
                        pygame.quit()
                        return
                    
                    for ui in ui_elements:
                        ui.handle_event(event)
                
                if should_back: 
                    done = True
                    break
                if should_reset:
                    done = True
                    break # Will handle reset at top of episode loop
                
                # Logic Steps based on Speed Slider
                sim_steps = int(slider_speed.value)
                if toggle_fast.checked:
                    sim_steps = 50 # Max speed
                    
                # We only render the LAST step of the batch to keep FPS stable
                for step_i in range(sim_steps):
                    if done: break
                    
                    # Timeout
                    if pygame.time.get_ticks() - start_time > 5000: # 5s timeout
                         for car in cars:
                            if car.alive:
                                car.alive = False
                                car.total_reward -= 1000
                         done = True
                         continue
                         
                    if all(not c.alive for c in cars):
                        done = True
                        continue
                        
                    # Agent Action
                    actions_chosen = []
                    for i, car in enumerate(cars):
                        if car.alive:
                            actions_chosen.append(agent.choose_action(states[i]))
                        else:
                            actions_chosen.append(0)
                            
                    frame_skip_rewards = [0] * NUM_CARS
                    
                    # Physics Step (Skip 4 frames logic preserved? Or just 1 step?)
                    # Original code skipped 4 frames. Let's do 1 physics step per loop for smoother control
                    # But to match original training speed, we might want to keep it.
                    # Let's do 1 step here for simplicity in new architecture.
                    
                    for i, car in enumerate(cars):
                        if not car.alive: continue
                        
                        action = actions_chosen[i]
                        if action == 1: car.accelerate()
                        elif action == 2: car.brake()
                        elif action == 3: car.steer_left()
                        elif action == 4: car.steer_right()
                        
                        car.update()
                        
                        # Collision
                        car_rect = pygame.Rect(car.x - car.width/2, car.y - car.length/2, car.width, car.length)
                        step_reward = -1
                        hit_wall = False
                        
                        # Check Track Bounds (0-800, 0-600)
                        if car.x < 0 or car.x > TRACK_WIDTH or car.y < 0 or car.y > TRACK_HEIGHT:
                            hit_wall = True
                        
                        for block in blocks:
                            if car_rect.colliderect(block): hit_wall = True
                            
                        if hit_wall:
                            step_reward -= 5000
                            car.alive = False
                            
                        if car_rect.colliderect(finish_line):
                            step_reward += 5000
                            car.alive = False
                            finished_count += 1
                            
                        # Shaping
                        step_reward += car.speed * 0.3
                        current_sensors = states[i][:-1]
                        if current_sensors[0] == 0: step_reward -= 5
                        if current_sensors[4] == 0: step_reward -= 5
                        
                        frame_skip_rewards[i] += step_reward
                        car.total_reward += step_reward
                        
                    # Learn
                    for i, car in enumerate(cars):
                        if actions_chosen[i] != 0:
                            next_state = get_state(car, blocks)
                            agent.learn(states[i], actions_chosen[i], frame_skip_rewards[i], next_state, not car.alive)
                            states[i] = next_state
                            
                    if agent_name == "DQN (NN)":
                        agent.train_step()
                        
                    frame_count += 1

                # --- RENDER ---
                # Always render UI to keep window responsive
                screen.fill(COLORS["BACKGROUND"])
                
                # 1. Sidebar
                sidebar_rect = pygame.Rect(0, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)
                pygame.draw.rect(screen, COLORS["PANEL_BG"], sidebar_rect)
                pygame.draw.line(screen, COLORS["BORDER"], (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, SCREEN_HEIGHT))
                
                # Sidebar Title
                font_h = get_font(20, bold=True)
                t = font_h.render("Control Panel", True, COLORS["TEXT_PRIMARY"])
                screen.blit(t, (20, 20))
                
                # Stats
                font_s = get_font(14)
                stats = [
                    f"Episode: {episode}",
                    f"Alive: {sum(1 for c in cars if c.alive)}",
                    f"Epsilon: {agent.epsilon:.2f}"
                ]
                for i, txt in enumerate(stats):
                    s = font_s.render(txt, True, COLORS["TEXT_SECONDARY"])
                    screen.blit(s, (20, 60 + i*20))

                for ui in ui_elements:
                    ui.draw(screen)
                    
                # 2. Viewport (Center)
                vp_rect = pygame.Rect(VIEWPORT_X, VIEWPORT_Y, VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
                pygame.draw.rect(screen, COLORS["BLACK"], vp_rect, border_radius=12)
                pygame.draw.rect(screen, COLORS["BORDER"], vp_rect, 1, border_radius=12)
                
                if not toggle_fast.checked:
                    # Center the track (800x600) in the viewport area
                    # Calculate offset to center track
                    offset_x = VIEWPORT_X + (VIEWPORT_WIDTH - TRACK_WIDTH) // 2
                    offset_y = VIEWPORT_Y + (VIEWPORT_HEIGHT - TRACK_HEIGHT) // 2
                    
                    # Draw Track
                    pygame.draw.line(screen, COLORS["BORDER"], (offset_x + 800, offset_y), (offset_x + 800, offset_y + 600), 2)
                    for block in blocks:
                        r = block.move(offset_x, offset_y)
                        pygame.draw.rect(screen, COLORS["BORDER"], r)
                    pygame.draw.rect(screen, COLORS["SUCCESS"], finish_line.move(offset_x, offset_y))
                    
                    # Draw Cars
                    best_car_idx = -1
                    max_reward = -float('inf')
                    for i, c in enumerate(cars):
                        if c.total_reward > max_reward:
                            max_reward = c.total_reward
                            best_car_idx = i
                            
                    for i, car in enumerate(cars):
                        if car.alive:
                            draw_car(screen, car, offset_x, offset_y, is_best=(i==best_car_idx))
                            
                    # Sensors (Optional)
                    if toggle_sensors.checked and best_car_idx != -1:
                        # Draw sensors for best car
                        pass
                else:
                    # Fast Mode Text
                    font_fast = get_font(32, bold=True)
                    fast_text = font_fast.render("FAST MODE ACTIVE", True, COLORS["ACCENT_BLUE"])
                    screen.blit(fast_text, (vp_rect.centerx - fast_text.get_width()//2, vp_rect.centery))

                # 3. Analysis Deck (Right)
                analysis_x = SCREEN_WIDTH - ANALYSIS_WIDTH
                
                # Graph Panel (Top)
                graph_rect = pygame.Rect(analysis_x + 10, 20, ANALYSIS_WIDTH - 20, 300)
                draw_graph(screen, graph_rect, avg_reward_history, completion_history)
                
                # Brain Panel (Bottom)
                if toggle_brain.checked:
                    brain_rect = pygame.Rect(analysis_x + 10, 340, ANALYSIS_WIDTH - 20, 300)
                    
                    # Viz update
                    current_state = None
                    current_action = None
                    
                    # Only update viz if not in fast mode OR if we want to see brain in fast mode?
                    # Usually fast mode implies no viz. But let's show it if requested.
                    # But finding best car might be expensive if we didn't do it above.
                    
                    if not toggle_fast.checked:
                         # We already found best_car_idx above
                         pass
                    else:
                        # Find best car for viz
                        best_car_idx = -1
                        max_reward = -float('inf')
                        for i, c in enumerate(cars):
                            if c.total_reward > max_reward:
                                max_reward = c.total_reward
                                best_car_idx = i
                    
                    if best_car_idx != -1:
                        if agent_name == "DQN (NN)":
                            v_state = get_state(cars[best_car_idx], blocks)
                            agent.forward(np.array(v_state))
                        elif agent_name == "Q-Table":
                            current_state = states[best_car_idx]
                            current_action = actions_chosen[best_car_idx]
                            
                    draw_network_or_heatmap(screen, brain_rect, agent, agent_name, heatmap, current_state, current_action)

                pygame.display.flip()
                clock.tick(FPS)
            
            # End Episode
            avg_reward = sum(c.total_reward for c in cars) / NUM_CARS
            avg_reward_history.append(avg_reward)
            completion_rate = (finished_count / NUM_CARS) * 100
            completion_history.append(completion_rate)
            agent.decay_epsilon()
            
            if should_back: break

    pygame.quit()

if __name__ == "__main__":
    main()
