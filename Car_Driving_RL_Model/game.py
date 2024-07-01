import pygame
import math
import time
import pickle
import os

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

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Racing Game")

# Load car image
car_image = pygame.image.load('Car_driving_game/Car_bit_art.png')
car_image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))

# Car parameters
car_speed = 0
car_angle = START_ANGLE
car_position = [START_X, START_Y]

# Timer variables
lap_start_time = None
best_lap_time = None
lap_completed = False

# Load best lap time if it exists
if os.path.exists('Car_driving_game/best_lap_time_user.pkl'):
    with open('Car_driving_game/best_lap_time_user.pkl', 'rb') as f:
        best_lap_time = pickle.load(f)

# Helper functions
def draw_track():
    pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), TRACK_OUTER_RADIUS, TRACK_WIDTH)

def check_on_track(pos):
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    dist = math.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
    return TRACK_OUTER_RADIUS - TRACK_WIDTH <= dist <= TRACK_OUTER_RADIUS

def draw_rotated_image(surface, image, center, angle):
    rotated_image = pygame.transform.rotate(image, angle * -1)
    new_rect = rotated_image.get_rect(center=image.get_rect(center=center).center)
    surface.blit(rotated_image, new_rect.topleft)

def calculate_progress_angle(start_pos, current_pos):
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    start_angle = math.atan2(center_y - start_pos[1], start_pos[0] - center_x)
    current_angle = math.atan2(center_y - current_pos[1], current_pos[0] - center_x)
    progress_angle = math.degrees(current_angle - start_angle)
    if progress_angle < 0:
        progress_angle += 360
    return progress_angle

def draw_car_info(speed, angle, lap_time, best_lap_time):
    font = pygame.font.Font(None, 36)
    speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
    position_text = font.render(f"Angle: {angle:.0f}°", True, WHITE)
    lap_time_text = font.render(f"Lap Time: {lap_time:.2f}s", True, WHITE)
    best_lap_time_text = font.render(f"Best Lap Time: {best_lap_time:.2f}s" if best_lap_time else "Best Lap Time: N/A", True, WHITE)

    screen.blit(speed_text, (10, 10))
    screen.blit(position_text, (10, 50))
    screen.blit(lap_time_text, (10, 90))
    screen.blit(best_lap_time_text, (10, 130))

def draw_arrow_keys(active_key):
    key_positions = [
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN + ARROW_KEY_SIZE + ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN + 2 * (ARROW_KEY_SIZE + ARROW_KEYS_MARGIN) - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN),
        (SCREEN_WIDTH - ARROW_KEYS_AREA_SIZE + ARROW_KEY_SIZE + 2 * ARROW_KEYS_MARGIN - 100, SCREEN_HEIGHT - ARROW_KEYS_AREA_SIZE + ARROW_KEYS_MARGIN - (ARROW_KEY_SIZE + ARROW_KEYS_MARGIN))
    ]
    
    for pos in key_positions:
        pygame.draw.rect(screen, WHITE, (pos[0] - 4, pos[1] - 4, ARROW_KEY_BACKGROUND_SIZE, ARROW_KEY_BACKGROUND_SIZE))
    
    key_colors = [ARROW_KEY_COLOR_INACTIVE] * 4
    
    if active_key == pygame.K_UP:
        key_colors[3] = ARROW_KEY_COLOR_ACTIVE
    elif active_key == pygame.K_DOWN:
        key_colors[1] = ARROW_KEY_COLOR_ACTIVE
    elif active_key == pygame.K_LEFT:
        key_colors[0] = ARROW_KEY_COLOR_ACTIVE
    elif active_key == pygame.K_RIGHT:
        key_colors[2] = ARROW_KEY_COLOR_ACTIVE
    
    for pos, color in zip(key_positions, key_colors):
        pygame.draw.rect(screen, color, (pos[0], pos[1], ARROW_KEY_SIZE, ARROW_KEY_SIZE))

# Game loop
running = True
clock = pygame.time.Clock()
active_key = None
progress_angle = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                active_key = event.key
        elif event.type == pygame.KEYUP:
            if event.key == active_key:
                active_key = None

    if car_speed == 0 and progress_angle == 0:
        lap_start_time = None
    else:
        if lap_start_time is None:
            lap_start_time = time.time()

    if active_key == pygame.K_UP:
        car_speed = min(car_speed + CAR_SPEED_INCREMENT, MAX_CAR_SPEED)
    elif active_key == pygame.K_DOWN:
        car_speed = max(car_speed - BRAKE_DECELERATION, 0)
    else:
        if car_speed > 0:
            car_speed = max(car_speed - FRICTION, 0)

    turn_rate = 3

    if car_speed > 0:
        if active_key == pygame.K_LEFT:
            car_angle += turn_rate
        elif active_key == pygame.K_RIGHT:
            car_angle -= turn_rate

    car_position[0] += car_speed * math.cos(math.radians(car_angle))
    car_position[1] -= car_speed * math.sin(math.radians(car_angle))

    progress_angle = calculate_progress_angle([START_X, START_Y], car_position)

    if not check_on_track(car_position):
        car_position = [START_X, START_Y]
        car_speed = 0
        car_angle = START_ANGLE
        lap_start_time = None

    # Check if a lap is completed
    if progress_angle >= 359:
        lap_completed = True

    if lap_completed:
        lap_time = time.time() - lap_start_time
        if best_lap_time is None or lap_time < best_lap_time:
            best_lap_time = lap_time
            with open('Car_driving_game/best_lap_time_user.pkl', 'wb') as f:
                pickle.dump(best_lap_time, f)
        car_position = [START_X, START_Y]
        car_speed = 0
        car_angle = START_ANGLE
        lap_start_time = None
        lap_completed = False

    lap_time = time.time() - lap_start_time if lap_start_time else 0

    screen.fill(BLACK)
    draw_track()
    draw_rotated_image(screen, car_image, car_position, -car_angle)
    draw_arrow_keys(active_key)
    draw_car_info(car_speed, progress_angle, lap_time, best_lap_time)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
