import pygame
import random
import numpy as np

class SnakeEnv:
    def __init__(self, width=720, height=480, block_size=10, difficulty=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.difficulty = difficulty

        self.game_window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake RL')

        self.clock = pygame.time.Clock()

        self.colors = {
            "black": pygame.Color(0, 0, 0),
            "white": pygame.Color(255, 255, 255),
            "red": pygame.Color(255, 0, 0),
            "green": pygame.Color(0, 255, 0),
        }

        self.reset()

    def reset(self):
        self.direction = 'RIGHT'
        self.snake_pos = [100, 50]
        self.snake_body = [self.snake_pos[:], [90, 50], [80, 50]]
        self.spawn_food()
        self.score = 0
        self.done = False
        return self.get_state()

    def spawn_food(self):
        while True:
            food_x = random.randrange(0, self.width // self.block_size) * self.block_size
            food_y = random.randrange(0, self.height // self.block_size) * self.block_size
            food_pos = [food_x, food_y]
            if food_pos not in self.snake_body:
                self.food_pos = food_pos
                break

    def step(self, action):
        # Actions: 0 = straight, 1 = right, 2 = left
        self.change_direction(action)
        self.move()

        reward = 0
        if self.snake_pos == self.food_pos:
            self.snake_body.insert(0, list(self.snake_pos))
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake_body.insert(0, list(self.snake_pos))
            self.snake_body.pop()

        # Check collisions
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height or
            self.snake_pos in self.snake_body[1:]):
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done, {}

    def change_direction(self, action):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)

        if action == 1:  # turn right
            self.direction = directions[(idx + 1) % 4]
        elif action == 2:  # turn left
            self.direction = directions[(idx - 1) % 4]
        # else action == 0: keep same direction

    def move(self):
        if self.direction == 'UP':
            self.snake_pos[1] -= self.block_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.block_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.block_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.block_size

    def get_state(self):
        #11-dim state: danger (front/right/left), direction, food direction
        head_x, head_y = self.snake_pos
        point_l = [head_x - self.block_size, head_y]
        point_r = [head_x + self.block_size, head_y]
        point_u = [head_x, head_y - self.block_size]
        point_d = [head_x, head_y + self.block_size]

        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'

        danger_straight = (dir_r and point_r in self.snake_body) or \
                          (dir_l and point_l in self.snake_body) or \
                          (dir_u and point_u in self.snake_body) or \
                          (dir_d and point_d in self.snake_body)

        danger_right = (dir_u and point_r in self.snake_body) or \
                       (dir_d and point_l in self.snake_body) or \
                       (dir_l and point_u in self.snake_body) or \
                       (dir_r and point_d in self.snake_body)

        danger_left = (dir_u and point_l in self.snake_body) or \
                      (dir_d and point_r in self.snake_body) or \
                      (dir_l and point_d in self.snake_body) or \
                      (dir_r and point_u in self.snake_body)

        food_left = self.food_pos[0] < self.snake_pos[0]
        food_right = self.food_pos[0] > self.snake_pos[0]
        food_up = self.food_pos[1] < self.snake_pos[1]
        food_down = self.food_pos[1] > self.snake_pos[1]

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down
        ], dtype=int)

        return state

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.game_window.fill(self.colors["black"])
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.colors["green"], pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        pygame.draw.rect(self.game_window, self.colors["red"], pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(self.difficulty)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    env.direction = 'UP'
                elif event.key == pygame.K_DOWN:
                    env.direction = 'DOWN'
                elif event.key == pygame.K_LEFT:
                    env.direction = 'LEFT'
                elif event.key == pygame.K_RIGHT:
                    env.direction = 'RIGHT'
                elif event.key == pygame.K_ESCAPE:
                    env.close()
                    exit()

        state, reward, done, _ = env.step(0)  # 0 = go straight (manual override via env.direction)
        env.render()

        if done:
            print(f"Score: {env.score}")
            pygame.time.wait(2000)
            state = env.reset()
