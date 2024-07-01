import pygame
import math
import random
import pickle
import os
import numpy as np
from collections import deque
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
TRACK_OUTER_RADIUS = 300
TRACK_WIDTH = 60
CAR_WIDTH = 40
CAR_HEIGHT = 30
CAR_SPEED_INCREMENT = 0.1
MAX_CAR_SPEED = 7
FRICTION = 0.02
BRAKE_DECELERATION = 0.3
START_ANGLE = 90
START_X = SCREEN_WIDTH // 2 + TRACK_OUTER_RADIUS - TRACK_WIDTH // 2
START_Y = SCREEN_HEIGHT // 2
ARROW_KEYS_AREA_SIZE = 120
ARROW_KEYS_MARGIN = 20
ARROW_KEY_SIZE = 50
ARROW_KEY_BACKGROUND_SIZE = ARROW_KEY_SIZE + 8
ARROW_KEY_COLOR_INACTIVE = (0, 0, 0)
ARROW_KEY_COLOR_ACTIVE = (255, 255, 255)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TRAINING_COLOR = (240, 46, 46)
TESTING_COLOR = (46, 240, 68)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Racing Game")

# Load car image
car_image = pygame.image.load('Car_Driving_RL_Model/Car_bit_art.png')
car_image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))

# Car parameters
car_speed = 0
car_angle = START_ANGLE
car_position = [START_X, START_Y]

# Mode: "train" or "test"
test_or_train = "train"
initial_mode = test_or_train

# Try lap while train flag
try_lap_while_train = False

# Q-learning parameters
alpha_base = 0.15  # Base learning rate or rate with static alpha
alpha_max = 0.2  # Maximum learning rate with dynamic alpha
alpha_min = 0.1 # Minimum learning rate with dynamic alpha
gamma = 0.99  # Discount factor
dynamic_vars = False # Chooses whether epsilon is static or dynamic along the track
epsilon_base = 0.2  # Base exploration rate or rate with static epsilon
epsilon_max = 0.1  # Maximum exploration rate with dynamic epsilon
epsilon_min = 0.005  # Minimum exploration rate with dynamic epsilon
proportion_before_explore = 0.5  # Proportion of the track where epsilon is 0 before exploration
actions = ["ACCELERATE", "BRAKE", "LEFT", "RIGHT", "NEUTRAL"]
q_table = {}

# Load Q-table, iteration counter, and average angle if they exist
q_table_path = 'Car_Driving_RL_Model/q_table_final.pkl' if initial_mode == "test" else 'Car_Driving_RL_Model/q_table.pkl'
if os.path.exists(q_table_path):
    with open(q_table_path, 'rb') as f:
        q_table = pickle.load(f)

if os.path.exists('Car_Driving_RL_Model/iterations.pkl'):
    with open('Car_Driving_RL_Model/iterations.pkl', 'rb') as f:
        iterations = pickle.load(f)
else:
    iterations = 0

if os.path.exists('Car_Driving_RL_Model/angles.pkl'):
    with open('Car_Driving_RL_Model/angles.pkl', 'rb') as f:
        angles = pickle.load(f)
else:
    angles = deque(maxlen=50)

if os.path.exists('Car_Driving_RL_Model/best_lap_time.pkl'):
    with open('Car_Driving_RL_Model/best_lap_time.pkl', 'rb') as f:
        best_lap_time = pickle.load(f)
else:
    best_lap_time = None

# Initialize avg_angle_reached
if len(angles) > 0:
    avg_angle_reached = sum(angles) / len(angles)
else:
    avg_angle_reached = 0

# Timer variables
lap_start_time = None
lap_completed = False
prev_angle = START_ANGLE

# Helper functions
def draw_track():
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), TRACK_OUTER_RADIUS, TRACK_WIDTH)

def check_on_track(pos):
    dist = calculate_distance(pos)
    return TRACK_OUTER_RADIUS - TRACK_WIDTH <= dist <= TRACK_OUTER_RADIUS

def calculate_distance(pos):
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    return math.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)

def draw_rotated_image(surface, image, center, angle):
    rotated_image = pygame.transform.rotate(image, angle * -1)
    new_rect = rotated_image.get_rect(center=image.get_rect(center=center).center)
    surface.blit(rotated_image, new_rect.topleft)

def get_state(car_speed, relative_car_angle_degrees, distance):
    return (round(car_speed, 1), round(relative_car_angle_degrees), round(distance))

def choose_action(state, current_angle):
    global epsilon
    if test_or_train == "test":
        epsilon = 0
    else:
        if not dynamic_vars:
            epsilon = epsilon_base
        else:
            if len(angles) > 0:
                avg_angle_reached = sum(angles) / len(angles)
                if current_angle < proportion_before_explore * avg_angle_reached:
                    epsilon = 0
                else:
                    epsilon = epsilon_min + (epsilon_max - epsilon_min) * ((current_angle / avg_angle_reached) ** 5)
                    epsilon = min(max(epsilon, epsilon_min), epsilon_max)
            else:
                epsilon = epsilon_base

    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        if state not in q_table:
            q_table[state] = {action: 0 for action in actions}
        return max(q_table[state], key=q_table[state].get)

def calculate_alpha(current_angle):
    if test_or_train == "test":
        return 0
    if not dynamic_vars:
        return alpha_base
    if len(angles) > 0:
        avg_angle_reached = sum(angles) / len(angles)
        alpha = alpha_min + (alpha_max - alpha_min) * ((current_angle / avg_angle_reached) ** 2)
        return min(max(alpha, alpha_min), alpha_max)
    else:
        return alpha_base

def update_q_table(state, action, reward, next_state, current_angle):
    alpha = calculate_alpha(current_angle)

    if state not in q_table:
        q_table[state] = {action: 0 for action in actions}
    if next_state not in q_table:
        q_table[next_state] = {action: 0 for action in actions}

    q_predict = q_table[state][action]
    q_target = reward + gamma * max(q_table[next_state].values())

    q_table[state][action] += alpha * (q_target - q_predict)

    if test_or_train == "train":
        with open('Car_Driving_RL_Model/q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)

def calculate_progress_angle(start_pos, current_pos):
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    start_angle = math.atan2(center_y - start_pos[1], start_pos[0] - center_x)
    current_angle = math.atan2(center_y - current_pos[1], current_pos[0] - center_x)
    progress_angle = current_angle - start_angle
    if progress_angle < 0:
        progress_angle += 2 * math.pi
    angle_degrees = math.degrees(progress_angle)
    if progress_angle < 0:
        progress_angle += 360
    return progress_angle, angle_degrees

def calculate_relative_car_angle_degrees(angle_degrees, car_angle):
    optimal_angle = (angle_degrees + 90) % 360
    relative_car_angle = car_angle - optimal_angle
    if relative_car_angle < -180:
        relative_car_angle += 360
    if relative_car_angle > 180:
        relative_car_angle -= 360
    return relative_car_angle

def normalize_reward(reward):
    return np.tanh(reward)

def draw_car_info(speed, angle, reward, epsilon, avg_angle, alpha, lap_time, best_lap_time):
    font = pygame.font.Font(None, 36)
    iterations_text = font.render(f"Iterations: {iterations}", True, WHITE)
    speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
    position_text = font.render(f"Angle: {angle:.0f}°", True, WHITE)
    reward_text = font.render(f"Reward: {reward:.2f}", True, WHITE)
    alpha_text = font.render(f"Alpha: {alpha:.2f}", True, WHITE)
    epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, WHITE)
    avg_angle_text = font.render(f"Avg Angle: {avg_angle:.1f}°", True, WHITE)

    lap_time_text = font.render(f"Lap Time: {lap_time:.2f}", True, WHITE)
    lap_time_seconds = font.render(f"s", True, WHITE)
    best_lap_text = font.render(f"Best Lap Time:", True, WHITE)
    best_lap_number = font.render(f"{best_lap_time:.2f}s" if best_lap_time else "N/A", True, WHITE)
    mode_text = font.render("Training mode" if test_or_train == "train" else "Testing mode", True, TRAINING_COLOR if test_or_train == "train" else TESTING_COLOR)
    
    screen.blit(iterations_text, (10, 10))
    screen.blit(speed_text, (10, 50))
    screen.blit(position_text, (10, 90))
    screen.blit(reward_text, (10, 130))
    screen.blit(alpha_text, (10, 170))
    screen.blit(epsilon_text, (10, 210))
    screen.blit(avg_angle_text, (10, 250))
    
    screen.blit(lap_time_text, (SCREEN_WIDTH - 250, 10))
    screen.blit(lap_time_seconds, (SCREEN_WIDTH - 70 if lap_time_text.get_width() < 175 else SCREEN_WIDTH - 60, 10))
    screen.blit(best_lap_text, (SCREEN_WIDTH - 250, 50))
    screen.blit(best_lap_number, (SCREEN_WIDTH - 250 + best_lap_text.get_width() + 5, 50))
    screen.blit(mode_text, (SCREEN_WIDTH - 250, 90))

def draw_arrow_keys(up, down, left, right):
    key_positions = [
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN + ARROW_KEY_SIZE + ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN + 2 * (ARROW_KEY_SIZE + ARROW_KEYS_MARGIN) - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEY_SIZE + 2 * ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN - (ARROW_KEY_SIZE + ARROW_KEYS_MARGIN))
    ]

    for pos in key_positions:
        pygame.draw.rect(screen, WHITE, (pos[0] - 4, pos[1] - 4, ARROW_KEY_BACKGROUND_SIZE, ARROW_KEY_BACKGROUND_SIZE))

    key_colors = [ARROW_KEY_COLOR_INACTIVE] * 4

    if up:
        key_colors[3] = ARROW_KEY_COLOR_ACTIVE
    if down:
        key_colors[1] = ARROW_KEY_COLOR_ACTIVE
    if left:
        key_colors[0] = ARROW_KEY_COLOR_ACTIVE
    if right:
        key_colors[2] = ARROW_KEY_COLOR_ACTIVE

    for pos, color in zip(key_positions, key_colors):
        pygame.draw.rect(screen, color, (pos[0], pos[1], ARROW_KEY_SIZE, ARROW_KEY_SIZE))

# Game loop
running = True
clock = pygame.time.Clock()
current_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if lap_start_time is None:
        lap_start_time = time.time()

    up, down, left, right = 0, 0, 0, 0
    progress_angle, angle_degrees = calculate_progress_angle([START_X, START_Y], car_position)
    relative_car_angle_degrees = calculate_relative_car_angle_degrees(angle_degrees, car_angle)
    dist = calculate_distance(car_position)
    state = get_state(car_speed, relative_car_angle_degrees, dist)
    action = choose_action(state, angle_degrees)

    if action == "ACCELERATE":
        car_speed = min(car_speed + CAR_SPEED_INCREMENT, MAX_CAR_SPEED)
        up = 1
    elif action == "BRAKE":
        car_speed = max(car_speed - BRAKE_DECELERATION, 0)
        down = 1
    elif action == "LEFT":
        car_angle += 3 if car_speed > 0 else 0
        left = 1
    elif action == "RIGHT":
        car_angle -= 3 if car_speed > 0 else 0
        right = 1
    elif action == "NEUTRAL":
        car_speed = max(car_speed - FRICTION, 0)

    car_position[0] += car_speed * math.cos(math.radians(car_angle))
    car_position[1] -= car_speed * math.sin(math.radians(car_angle))
    progress_angle, angle_degrees = calculate_progress_angle([START_X, START_Y], car_position)
    relative_car_angle_degrees = calculate_relative_car_angle_degrees(angle_degrees, car_angle)
    dist = calculate_distance(car_position)

    if check_on_track(car_position):
        angle_change = angle_degrees - prev_angle
        raw_reward = angle_change * max(1,car_speed ** 2) if angle_change > 0 else 0
    else:
        raw_reward = -20

    current_reward = raw_reward
    prev_angle = angle_degrees

    # Check if a lap is completed
    if angle_degrees >= 359:
        lap_completed = True

    next_state = get_state(car_speed, relative_car_angle_degrees, dist)
    if test_or_train == "train":
        update_q_table(state, action, current_reward, next_state, angle_degrees)

    if not check_on_track(car_position) or lap_completed:
        if lap_completed:
            lap_time = time.time() - lap_start_time
            if best_lap_time is None or lap_time < best_lap_time:
                best_lap_time = lap_time
                with open('Car_Driving_RL_Model/best_lap_time.pkl', 'wb') as f:
                    pickle.dump(best_lap_time, f)
                # Save the best q_table
                with open('Car_Driving_RL_Model/q_table_final.pkl', 'wb') as f:
                    pickle.dump(q_table, f)
        if test_or_train == "train":
            if dynamic_vars:
                angles.append(angle_degrees)
                avg_angle_reached = sum(angles) / len(angles)

            iterations += 1
            with open('Car_Driving_RL_Model/angles.pkl', 'wb') as f:
                pickle.dump(angles, f)

            with open('Car_Driving_RL_Model/iterations.pkl', 'wb') as f:
                pickle.dump(iterations, f)

        car_position = [START_X, START_Y]
        car_speed = 0
        car_angle = START_ANGLE
        lap_start_time = None
        lap_completed = False

        if try_lap_while_train and test_or_train == "train":
            test_or_train = "test"
        elif initial_mode == "train":
            test_or_train = "train"
            if lap_completed:
                test_or_train = "test"
                lap_completed = False

    alpha = calculate_alpha(angle_degrees)
    lap_time = time.time() - lap_start_time if lap_start_time else 0

    screen.fill(BLACK)
    draw_track()
    draw_rotated_image(screen, car_image, car_position, -car_angle)
    draw_car_info(car_speed, angle_degrees, current_reward, epsilon, avg_angle_reached, alpha, lap_time, best_lap_time)
    draw_arrow_keys(up, down, left, right)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
