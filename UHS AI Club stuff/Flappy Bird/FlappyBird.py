import pygame
import sys
import random
import torch
import torch.nn as nn
#for renforcement learning
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


#initialize the neural network

class flapAI(nn.Module): 
    def __init__(self, input_size, output_size):
        super(flapAI, self).__init__()
        self.x1 = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.Sigmoid(),
            nn.Linear(10,output_size)
        )
    
    def forward(self, x):
        x = self.x1(x)
        x = F.softmax(x, dim=1)
        return x
    
#initialize the neural network
#input size is 4 because the input is the bird's y position, the bird's y velocity, the distance to the next pipe, and the height of the next pipe
input_size = 4
#output size is 2 because the output is either flap or not flap
output_size = 2


#RQL algorithm
class RQL:
    def __init__(self, input_size, output_size):
        self.model = flapAI(input_size, output_size)
        self.memory = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.gamma = 0.9
        
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, input_size)
        return self.model(state).argmax().item()
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def experience_replay(self):
        if len(self.memory) < 100:
            return
        batch = random.sample(self.memory, 100)
        for state, action, reward, next_state in batch:
            state = torch.tensor(state, dtype=torch.float).view(1, input_size)
            next_state = torch.tensor(next_state, dtype=torch.float).view(1, input_size)
            reward = torch.tensor(reward, dtype=torch.float).view(1, 1)
            target = reward + self.gamma * self.model(next_state).max().item()
            target = torch.tensor(target, dtype=torch.float).view(1, 1)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state), target)
            loss.backward()
            self.optimizer.step()

#initialize the RQL
rql = RQL(input_size, output_size)


# Initialize Pygame
pygame.init() 

# Screen dimensions
SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 600




# Create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Load the background image
background_image = pygame.image.load("C:/Users/Temp/Documents/ML/UHS AI Club stuff/Flappy Bird/background.png").convert()
background_image = pygame.transform.scale(background_image, (800, 600))

# Clock object to control the frame rate
clock = pygame.time.Clock()

# Load other images
bird_image = pygame.image.load("C:/Users/Temp/Documents/ML/UHS AI Club stuff/Flappy Bird/bird.png").convert_alpha()

# Modify the pipe image to make it thinner
original_pipe_image = pygame.image.load("C:/Users/Temp/Documents/ML/UHS AI Club stuff/Flappy Bird/pipe.png").convert_alpha()
pipe_width = int(original_pipe_image.get_width() * 0.5)  # Reduce width to 50% of original
pipe_height = original_pipe_image.get_height()
pipe_image = pygame.transform.scale(original_pipe_image, (pipe_width, pipe_height))

# Bird class
class Bird:
    def __init__(self):
        self.image = pygame.transform.scale(bird_image, (bird_image.get_width() // 7, bird_image.get_height() // 7 )) 
        self.x = 40
        self.y = 20
        self.velocity = 0
        self.gravity = 0.5
 
    def flap(self):
        self.velocity = -8

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        # Prevent the bird from going off-screen
        if self.y <= 0:
            self.y = 0
            self.velocity = 0
        if self.y + self.image.get_height() >= SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.image.get_height()
            self.velocity = 0

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap = 200
        self.image = pipe_image
        self.width = self.image.get_width()
        self.top_height = random.randint(50, SCREEN_HEIGHT - self.gap - 50 )
        self.bottom_height = SCREEN_HEIGHT - self.top_height - self.gap

    def update(self):
        self.x -= 3

    def draw(self, surface):
        # Top pipe (flipped)
        top_pipe_image = pygame.transform.flip(self.image, False, True)
        surface.blit(top_pipe_image, (self.x, self.top_height - self.image.get_height()))
        # Bottom pipe
        surface.blit(self.image, (self.x, SCREEN_HEIGHT - self.bottom_height))

    def collide(self, bird):
        bird_rect = pygame.Rect(bird.x, bird.y,60,50)
        # Top pipe rect
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        # Bottom pipe rect
        bottom_pipe_rect = pygame.Rect(self.x, SCREEN_HEIGHT - self.bottom_height, self.width, self.bottom_height)
        return bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect)

# Game variables
bird = Bird()
pipes = []
pipe_timer = 0

# Main game loop
running = True
while running:
    clock.tick(60)  # 60 FPS

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.flap()

    # Update game state
    bird.update()

    # Pipe generation
    pipe_timer += 1
    if pipe_timer >= 150:  # Increased from 90 to 150 for further spacing
        pipes.append(Pipe(SCREEN_WIDTH))
        pipe_timer = 0



    # Draw everything
    screen.blit(background_image, (0, 0))
    bird.draw(screen)
    for pipe in pipes:
        pipe.draw(screen)

    # Draw hitboxes
    bird_rect = pygame.Rect(bird.x, bird.y, 60,50)
    pygame.draw.rect(screen, (255, 0, 0), bird_rect, 2)
    for pipe in pipes:
        top_pipe_rect = pygame.Rect(pipe.x, 0, pipe.width, pipe.top_height-35)
        bottom_pipe_rect = pygame.Rect(pipe.x, SCREEN_HEIGHT - pipe.bottom_height+35, pipe.width, pipe.bottom_height)
        pygame.draw.rect(screen, (255, 0, 0), top_pipe_rect, 2)
        pygame.draw.rect(screen, (255, 0, 0), bottom_pipe_rect, 2)

        

        pipe.update()
        # Collision detection
        if top_pipe_rect.colliderect (bird_rect):
            running = False
        if bottom_pipe_rect.colliderect(bird_rect):
            running = False
        # Remove off-screen pipes
        if pipe.x + pipe.width < 0:
            pipes.remove(pipe) 

    #display a velocity vector and show the number
    pygame.draw.line(screen, (0, 255, 0), (bird.x + bird.image.get_width() // 2, bird.y + bird.image.get_height() // 2),
     (bird.x + bird.image.get_width() // 2, bird.y + bird.image.get_height() // 2 + bird.velocity * 10), 2)
    font = pygame.font.Font(None, 36)
    text = font.render(f"{bird.velocity}", 1, (0, 255, 0))
    screen.blit(text, (bird.x + bird.image.get_width() // 2, bird.y + bird.image.get_height() // 2 + bird.velocity*10 ))
           

    #display a red line from bird to next closest pipe
    if pipes:
        next_pipe = pipes[0]
        pygame.draw.line(screen, (255, 0, 0), (bird.x + bird.image.get_width() // 2, bird.y + bird.image.get_height() // 2), (next_pipe.x + pipe_width // 2, bird.y + bird.image.get_height() // 2), 2)
        # Display the distance to the next pipe
        font = pygame.font.Font(None, 36)
        distance_to_next_pipe = next_pipe.x - bird.x
        text = font.render(f"{distance_to_next_pipe}", 1, (255, 0, 0))
        screen.blit(text, (bird.x + bird.image.get_width() // 2 + 50, bird.y + bird.image.get_height() // 2))

    #display the height of the bird and a line to show it from the floor
    font = pygame.font.Font(None, 36)
    text = font.render(f"{bird.y}", 1, (0, 0, 255))
    screen.blit(text, (bird.x + bird.image.get_width() // 2 - 50, bird.y - 50))
    pygame.draw.line(screen, (0, 0, 255), (bird.x + bird.image.get_width() // 2, bird.y + bird.image.get_height() // 2), (bird.x + bird.image.get_width() // 2, SCREEN_HEIGHT), 2)



    #display height of next bottom and top pipe and a line to show it from the floor    
    if pipes:
        next_pipe = pipes[0]
         
        text = font.render(f"{next_pipe.bottom_height}", 1, (0, 0, 255))
        screen.blit(text, (next_pipe.x + pipe_width // 2, SCREEN_HEIGHT - next_pipe.bottom_height))
        text = font.render(f"{next_pipe.top_height}", 1, (0, 0, 255))
        screen.blit(text, (next_pipe.x + pipe_width // 2, next_pipe.top_height))
        # Draw a line to show the bottom pipe from the floor


        

        pygame.draw.line(screen, (0, 0, 255), (next_pipe.x + pipe_width // 2, SCREEN_HEIGHT - next_pipe.bottom_height+35), (next_pipe.x + pipe_width // 2, SCREEN_HEIGHT), 2)
        # and a line to show the top pipe from the ceiling
        pygame.draw.line(screen, (0, 0, 255), (next_pipe.x + pipe_width // 2, 0), (next_pipe.x + pipe_width // 2, next_pipe.top_height-35), 2)
        # Use the rect rights of the pipes

        pygame.draw.rect(screen, (0, 255, 0), top_pipe_rect, 2)
        pygame.draw.rect(screen, (0, 255, 0), bottom_pipe_rect, 2)

    


 


    # Display bird x, y, y-velo
    font = pygame.font.Font(None, 36)
    # Find the distance to the next pipe
    if pipes: 
        next_pipe = pipes[0]
        distance_to_next_pipe = next_pipe.x - bird.x
    else:
        distance_to_next_pipe = 0

    # Render the text
    text = font.render(f"x: {bird.x} y: {bird.y} y-velo: {bird.velocity} dist: {distance_to_next_pipe}", 1, (255, 255, 255))
    screen.blit(text, (10, 10))

    #draw rectangle at the right of the screen fill color white
    pygame.draw.rect(screen, (242, 228, 187), (SCREEN_WIDTH -500, 0, 500, SCREEN_HEIGHT))


    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
 