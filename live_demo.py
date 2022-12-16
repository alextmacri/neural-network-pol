import pygame as pg
import numpy as np
from live_demo_network import NeuralNetwork


# class to make drawing squares more efficient
class SquareColourFloat(float):
    """default value is 0.0"""
    def __add__(self, __x: float) -> float:
        # Implementation note: I have to convert to float to avoid recursion
        if float(self) + __x > 1:   # caps out at 1
            return SquareColourFloat(1.0)
        return SquareColourFloat(float(self) + __x)


# square drawing functions
grayscale = lambda x: (255 * x, 255 * x, 255 * x)

def colour_squares(x_cor: int, y_cor: int, data_grid: list[list[SquareColourFloat]], square_grid: list[list[pg.Rect]]) -> None:
    DRAW_INTENSITY_INNER = 0.75
    DRAW_INTENSITY_OUTER = 0.15

    i = y_cor * 28 // 800
    j = x_cor * 28 // 800

    data_grid[i][j] += DRAW_INTENSITY_INNER
    pg.draw.rect(screen, grayscale(data_grid[i][j]), square_grid[i][j])

    for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        offset_i = i + offset[0]
        offset_j = j + offset[1]
        if 0 <= offset_i < 28 and 0 <= offset_j < 28:
            data_grid[offset_i][offset_j] += DRAW_INTENSITY_OUTER
            pg.draw.rect(screen, grayscale(data_grid[offset_i][offset_j]), square_grid[offset_i][offset_j])


def render_all_squares(data_grid: list[list[SquareColourFloat]], square_grid: list[list[pg.Rect]]) -> None:
    for i in range(len(square_grid)):
        for j in range(len(square_grid[i])):
            pg.draw.rect(screen, grayscale(data_grid[i][j]), square_grid[i][j])


# text updating function
def update_text(data_grid: list[list[SquareColourFloat]]):
    """generates a guess based on the drawn stuff and updates the sidebar info text"""
    data = np.asarray(data_grid).reshape(784, 1)
    results = NN.forward_propagation(data)
    
    pg.draw.rect(screen, (0, 0, 0), text_background)

    for i in range(10):
        txt = font.render('{}: {:.1f}%'.format(i, results[i][0] * 100), True, (255, 255, 255))
        screen.blit(txt, (900, 20 + (i * 55)))

    txt = font.render(f'Final guess: {results.argmax()}', True, (255, 255, 255))
    screen.blit(txt, (830, 580))


# setting up neural network
NN = NeuralNetwork(784, 20, 10)
NN.load_from_file('live_demo_weights.npy', 'live_demo_biases.npy')


# setting up pygame window and drawing area
data_grid = [[SquareColourFloat() for _ in range(28)] for _ in range(28)]
square_size = 800 / 28 - 1
square_grid = [[pg.Rect(cx * 800 / 28, cy * 800 / 28, square_size, square_size) for cx in range(28)] for cy in range(28)]

screen = pg.display.set_mode((1200, 800))
screen.fill((30, 50, 30))
render_all_squares(data_grid, square_grid)


# setting up pygame info sidebar
pg.font.init()
font = pg.font.SysFont(None, 72)

clear_button = pg.Rect(801, 660, 399, 140)
pg.draw.rect(screen, (0, 0, 0), clear_button)
clear_txt = font.render('Clear', True, (255, 255, 255))
screen.blit(clear_txt, (950, 710))

text_background = pg.Rect(801, 0, 399, 657)
update_text(data_grid)


# pygame main loop
try:
    draw_on = False

    while True:
        e = pg.event.wait()
        if e.type == pg.QUIT:
            raise StopIteration

        if e.type == pg.MOUSEBUTTONDOWN:
            if pg.mouse.get_pos()[0] < 800:
                colour_squares(pg.mouse.get_pos()[0], pg.mouse.get_pos()[1], data_grid, square_grid)
                draw_on = True

            elif pg.mouse.get_pos()[1] > 660:
                data_grid = [[SquareColourFloat() for _ in range(28)] for _ in range(28)]
                render_all_squares(data_grid, square_grid)

        if e.type == pg.MOUSEBUTTONUP:
            draw_on = False
            update_text(data_grid)

        if e.type == pg.MOUSEMOTION:
            if draw_on and pg.mouse.get_pos()[0] < 800:
                colour_squares(pg.mouse.get_pos()[0], pg.mouse.get_pos()[1], data_grid, square_grid)

        pg.display.flip()
except StopIteration:
    pass

pg.quit()