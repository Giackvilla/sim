import pygame
from car import Car
from agent import QLearningAgent
from agent2 import DQNAgent
from game import get_state, draw_car, show_menu, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GREEN, RED

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Car ML - Trained Model Run")
    clock = pygame.time.Clock()

    # Show Menu
    agent_choice = show_menu(screen)
    if agent_choice is None: return

    # Track Elements (Same as game.py)
    blocks = [
        pygame.Rect(100, 0, 20, SCREEN_HEIGHT),   
        pygame.Rect(680, 0, 20, SCREEN_HEIGHT),   
    ]
    finish_line = pygame.Rect(120, 50, 560, 20)

    # Agent Setup
    actions = [0, 1, 2, 3, 4]
    
    if agent_choice == "classic":
        agent = QLearningAgent(actions)
        agent.load("q_table.pkl")
    else:
        agent = DQNAgent(actions)
        try:
            agent.load("dqn_model.pkl")
        except:
            print("No DQN model found!")
            return

    agent.epsilon = 0 # No exploration, pure exploitation
    
    episodes = 10
    
    for episode in range(episodes):
        car = Car(400, SCREEN_HEIGHT - 100, angle=0)
        state = get_state(car, blocks)
        
        done = False
        total_reward = 0
        
        print(f"Starting Episode {episode}...")
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Choose Action (Greedy)
            action = agent.choose_action(state)
            
            # Frame Skip Loop (Visual only, no learning)
            for _ in range(4):
                if done: break
                
                if action == 1: car.accelerate()
                elif action == 2: car.brake()
                elif action == 3: car.steer_left()
                elif action == 4: car.steer_right()
                
                car.update()
                
                car_rect = pygame.Rect(car.x - car.width/2, car.y - car.length/2, car.width, car.length)
                
                # Check blocks
                if car.x < 0 or car.x > 800 or car.y < 0 or car.y > SCREEN_HEIGHT:
                    print("Crashed into bounds!")
                    done = True
                
                for block in blocks:
                    if car_rect.colliderect(block):
                        print("Crashed into wall!")
                        done = True
                        break
                
                if car_rect.colliderect(finish_line):
                    print("FINISHED!")
                    done = True
                
                # Draw
                screen.fill(WHITE)
                pygame.draw.line(screen, BLACK, (800, 0), (800, SCREEN_HEIGHT), 2)
                for block in blocks:
                    pygame.draw.rect(screen, BLACK, block)
                pygame.draw.rect(screen, GREEN, finish_line)
                
                draw_car(screen, car, is_best=True)
                
                # Info
                font = pygame.font.SysFont(None, 36)
                info = f"Trained Run - Ep: {episode}"
                screen.blit(font.render(info, True, BLACK), (10, 10))
                
                pygame.display.flip()
                clock.tick(60) # Run at normal speed for viewing

            state = get_state(car, blocks)

    pygame.quit()

if __name__ == "__main__":
    main()
