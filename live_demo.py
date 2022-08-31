import pygame as pg
import pygame_widgets as pw
import numpy as np
from live_demo_network import NeuralNetwork

# setting up neural network
NN = NeuralNetwork(784, 20, 10)
NN.load_from_file('live_demo_weights.npy', 'live_demo_biases.npy')

# setting up pygame window and drawing area
screen = pg.display.set_mode((1200, 800))
pg.font.init()

draw_on = False
color = lambda x: (255*x, 255*x, 255*x)

data_grid = np.zeros((28, 28))
square_size = 800/28 - 1
square_grid = [[pg.Rect(cx*800/28, cy*800/28, square_size, square_size) for cx in range(28)] for cy in range(28)]

# pygame main loop
def colour_squares(x_cor, y_cor):
    adjustment_inner = 0.75
    adjustment_outer = 0.15

    i = int(y_cor*28/800)
    j = int(x_cor*28/800)

    if data_grid[i][j] + adjustment_inner > 1:
        data_grid[i][j] = 1
    else:
        data_grid[i][j] += adjustment_inner
    pg.draw.rect(screen, color(data_grid[i][j]), square_grid[i][j])
    
    if i-1 >= 0:
        if data_grid[i-1][j] + adjustment_outer > adjustment_inner:
            data_grid[i-1][j] = 1
        else:
            data_grid[i-1][j] += adjustment_outer
        pg.draw.rect(screen, color(data_grid[i-1][j]), square_grid[i-1][j])

    if i+1 <= 27:
        if data_grid[i+1][j] + adjustment_outer > adjustment_inner:
            data_grid[i+1][j] = 1
        else:
            data_grid[i+1][j] += adjustment_outer
        pg.draw.rect(screen, color(data_grid[i+1][j]), square_grid[i+1][j])

    if j-1 >= 0:
        if data_grid[i][j-1] + adjustment_outer > adjustment_inner:
            data_grid[i][j-1] = 1
        else:
            data_grid[i][j-1] += adjustment_outer
        pg.draw.rect(screen, color(data_grid[i][j-1]), square_grid[i][j-1])
    
    if j+1 <= 27:
        if data_grid[i][j+1] + adjustment_outer > adjustment_inner:
            data_grid[i][j+1] = 1
        else:
            data_grid[i][j+1] += adjustment_outer
        pg.draw.rect(screen, color(data_grid[i][j+1]), square_grid[i][j+1])

def render_all_squares():
    for i in range(len(square_grid)):
        for j in range(len(square_grid[i])):
            pg.draw.rect(screen, color(data_grid[i][j]), square_grid[i][j])

screen.fill((30, 50, 30))
render_all_squares()

# setting up pygame info sidebar
font = pg.font.SysFont(None, 48)

guess_button = pg.Rect(801, 660, 399, 140)
pg.draw.rect(screen, (0, 0, 0), guess_button)
guess_txt = font.render('Generate Guess', True, (255, 255, 255))
screen.blit(guess_txt, (870, 710))

clear_button = pg.Rect(801, 550, 399, 107)
pg.draw.rect(screen, (0, 0, 0), clear_button)
guess_txt = font.render('Clear', True, (255, 255, 255))
screen.blit(guess_txt, (950, 590))

def default_text():
    for i in range(10):
        txt = font.render(f'{i}:', True, (255, 255, 255))
        screen.blit(txt, (950, 20 + (i * 45)))
    txt = font.render(f'Final guess:', True, (255, 255, 255))
    screen.blit(txt, (900, 490))

def update_text(results):
    pg.draw.rect(screen, (0, 0, 0), text_background)
    for i in range(10):
        txt = font.render('{}: {:.3f}'.format(i, results[i][0]), True, (255, 255, 255))
        screen.blit(txt, (950, 20 + (i * 45)))
    txt = font.render(f'Final guess: {results.argmax()}', True, (255, 255, 255))
    screen.blit(txt, (900, 490))

text_background = pg.Rect(801, 0, 399, 547)
default_text()

# pygame main loop
try:
    while True:
        e = pg.event.wait()
        if e.type == pg.QUIT:
            raise StopIteration
        if e.type == pg.MOUSEBUTTONDOWN:
            if pg.mouse.get_pos()[0] < 800:
                colour_squares(pg.mouse.get_pos()[0], pg.mouse.get_pos()[1])
                draw_on = True
            elif pg.mouse.get_pos()[1] > 660:
                update_text(NN.forward_propagation(data_grid.reshape(784, 1)))
            elif pg.mouse.get_pos()[1] > 550:
                data_grid = np.zeros((28, 28))
                render_all_squares()
        if e.type == pg.MOUSEBUTTONUP:
            draw_on = False
        if e.type == pg.MOUSEMOTION:
            if draw_on and pg.mouse.get_pos()[0] < 800:
                colour_squares(pg.mouse.get_pos()[0], pg.mouse.get_pos()[1])

        pg.display.flip()

except StopIteration:
    pass

pg.quit()