import json
import os

class TrainingConfig:
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon_start = 1.0
        
        # Reward Weights
        self.goal_reward = 5000.0
        self.wall_penalty = -5000.0
        self.idle_penalty = -5.0 # For speed < 1
        self.sensor_penalty = -5.0 # For getting too close to walls
        self.speed_reward_factor = 0.3 # Multiplier for speed
        self.distance_reward_factor = 0.1 # Reward for getting closer to finish
        self.steering_penalty = -0.1 # Penalty for steering action
        
        # Toggles
        self.use_speed_reward = True
        self.use_wall_penalty = True
        self.use_idle_penalty = True
        self.use_sensor_penalty = True
        self.use_distance_reward = False
        self.use_steering_penalty = False
        self.use_goal_reward = True

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_start": self.epsilon_start,
            "goal_reward": self.goal_reward,
            "wall_penalty": self.wall_penalty,
            "idle_penalty": self.idle_penalty,
            "sensor_penalty": self.sensor_penalty,
            "speed_reward_factor": self.speed_reward_factor,
            "distance_reward_factor": self.distance_reward_factor,
            "steering_penalty": self.steering_penalty,
            "use_speed_reward": self.use_speed_reward,
            "use_wall_penalty": self.use_wall_penalty,
            "use_idle_penalty": self.use_idle_penalty,
            "use_sensor_penalty": self.use_sensor_penalty,
            "use_distance_reward": self.use_distance_reward,
            "use_steering_penalty": self.use_steering_penalty,
            "use_goal_reward": self.use_goal_reward
        }

    def from_dict(self, data):
        self.learning_rate = data.get("learning_rate", 0.1)
        self.discount_factor = data.get("discount_factor", 0.9)
        self.epsilon_start = data.get("epsilon_start", 1.0)
        self.goal_reward = data.get("goal_reward", 5000.0)
        self.wall_penalty = data.get("wall_penalty", -5000.0)
        self.idle_penalty = data.get("idle_penalty", -5.0)
        self.sensor_penalty = data.get("sensor_penalty", -5.0)
        self.speed_reward_factor = data.get("speed_reward_factor", 0.3)
        self.distance_reward_factor = data.get("distance_reward_factor", 0.1)
        self.steering_penalty = data.get("steering_penalty", -0.1)
        self.use_speed_reward = data.get("use_speed_reward", True)
        self.use_wall_penalty = data.get("use_wall_penalty", True)
        self.use_idle_penalty = data.get("use_idle_penalty", True)
        self.use_sensor_penalty = data.get("use_sensor_penalty", True)
        self.use_distance_reward = data.get("use_distance_reward", False)
        self.use_steering_penalty = data.get("use_steering_penalty", False)
        self.use_goal_reward = data.get("use_goal_reward", True)

    def save(self, filename="training_config.json"):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, filename="training_config.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                self.from_dict(data)
        else:
            print(f"Config file {filename} not found, using defaults.")

# Presets
def get_default_config():
    return TrainingConfig()

def get_aggressive_config():
    c = TrainingConfig()
    c.learning_rate = 0.2
    c.speed_reward_factor = 1.0
    c.wall_penalty = -1000.0 # Care less about crashing
    c.sensor_penalty = 0.0 # Don't fear walls
    c.use_sensor_penalty = False 
    c.use_steering_penalty = False
    return c

def get_cautious_config():
    c = TrainingConfig()
    c.learning_rate = 0.05
    c.wall_penalty = -10000.0 # Hitting wall is death
    c.sensor_penalty = -20.0 # Stay away!
    c.speed_reward_factor = 0.1 # Speed matters less
    c.use_distance_reward = True # Focus on completion
    c.use_steering_penalty = True # Drive smooth
    return c

PRESETS = {
    "Default": get_default_config,
    "Aggressive": get_aggressive_config,
    "Cautious": get_cautious_config
}
