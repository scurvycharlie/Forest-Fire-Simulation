#script for the running of basic forest fire. 

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import random

def retrieve_neighbouring_cells(grid, x, y):
    """
    The retrieve neighbouring cells function retrieves the cells 1 above, below, to the right and left of the target cell. 
    The function iterates over every cell in the grid, takes the indexes and fills the index_cells array
    """
    rows, cols = grid.shape #define grid shape as the number of rows and cols
    neighbouring_cells = [] #creating an empty array to fill with the neighbouring coordinates
    
    if x > 0: #if x coordinate isn't the 0 index
        neighbouring_cells.append(grid[x - 1, y]) #collect the cell upwards of it
    if x < rows - 1: #if the cell is not in the last row
        neighbouring_cells.append(grid[x + 1, y]) #collect the cell below it
    if y > 0: #if the column is not the 0 index
        neighbouring_cells.append(grid[x, y - 1]) #collect the cell to the left (-1)
    if y < cols - 1: #if the col is not the last col
        neighbouring_cells.append(grid[x, y + 1]) #collect the cell to the right (+1)
    
    return neighbouring_cells

def fire_check(neighbouring_cells):
    """
    The check fire function checks the neighbouring cells of the target cell for a 2 value, indicating fire.
    If the fire is present, the function returns true. 
    Otherwise the function returns false.
    """
    for n in neighbouring_cells: #any cell in the neighbouring cells
        if n == 2:
            return True #if any cell in the neighbouring cells is 2, then the function returns TRUE
    return False #if no 2 is present the function returns FALSE

def update_state(grid, probability_of_new_growth, probability_of_lightning_strike):
    grid_new = np.copy(grid) 
    rows, cols = grid.shape #define grid shape as the number of rows and cols
    
    for x in range(rows):
        for y in range(cols):
            target_cell = grid[x, y] #defining the target cell
            neighbouring_cells = retrieve_neighbouring_cells(grid, x, y) #retrieving the neighbouring cells
            FIRE = fire_check(neighbouring_cells) #if the fire_check function returns true
            
            if target_cell == 1 and FIRE:
                grid_new[x, y] = 2 #if FIRE = TRUE while target cell = 1 then cell updates to 2 (on fire)
                
            if target_cell == 2:
                grid_new[x, y] = 0 #cells on fire update to 0 (burnt out)
            
            #the np.random modules function rand() produces a random float from 0-1. This is used to simulate the probability of lightning ot growth.
            if target_cell == 0 and np.random.random() < probability_of_new_growth: #if rand() < prob while there is a tree present, then a fire appears
                grid_new[x, y] = 1 #
            
            if target_cell == 1 and np.random.random() < probability_of_lightning_strike:
                grid_new[x, y] = 2
                
    return grid_new

def run_simulation(forest, num_steps):
    forest_state = [forest]  # state is a list which stores the state of the forest for each step
    for step in range(num_steps): 
        forest = update_state(forest, probability_of_new_growth, probability_of_lightning_strike)
        forest_state.append(forest) #for loop runs the update_state on the returned forest for each step defined by num_steps
    return forest_state #return the stored simulated states


rows = 100 #define the number of rows (x)
cols = 100 #define the number of columns (y)
num_steps = 100 #100 steps

probability_of_new_growth = 0.01 #p value of a 0 becoming a 1 in the next time step
probability_of_lightning_strike = 0.002 #p value of 1 becoming a 2 in the next step

forest = np.random.choice([0, 1], size=(rows, cols))  #defining the forest size and the contents
forest[50, 50] = 2

sim1 = run_simulation(forest, num_steps) #defining the simulation for the steady state investigation

colours = ListedColormap(['#653700', '#008000', '#F97306']) #setting 0 as brown, 1 as green and 2 as orange. 

fig, ax = plt.subplots()
plot = ax.imshow(sim1[0], cmap = colours) #plotting to check the initial state is correct

from IPython.display import HTML

def animate(i):
    """
    amimation funtion, takes each iteration of sim1 and returns the plot, the funcAnimation function is used to visualise
    """
    plot.set_data(sim1[i])
    return [plot]

anim = FuncAnimation(fig, animate, frames=len(sim1), interval=200, blit=False) #animating the simulation

#HTML(anim.to_jshtml())
#plt.show()

anim.save('base_forest_fire_simulation.gif', writer='pillow') #saving animation output in gif format
    
    
#now the steady state is being investigated 


rows = 100 #redefining all values for steady state investigation
cols = 100
num_steps =  500 #steps to investigate the steady state

probability_of_new_growth = 0.07 #7% of regrowth i a burnt out cell
probability_of_lightning_strike = 0.0001 #0.01% chance of lightning striking

forest = np.random.choice([0, 1], size=(rows, cols))  #defining the forest size and the contents
forest[50, 50] = 2

sim_steady = run_simulation(forest, num_steps) #defining the simulation for the steady state investigation

colours = ListedColormap(['#653700', '#008000', '#F97306']) #setting 0 as brown, 1 as green and 2 as orange. 

def animate_steady(i):
    """
    amimation funtion, takes each iteration of sim1 and returns the plot, the funcAnimation function is used to visualise
    """
    plot.set_data(sim_steady[i])
    return [plot]

anim = FuncAnimation(fig, animate, frames=len(sim_steady), interval=200, blit=False)


anim.save('base_forest_fire_simulation_steadystate.gif', writer='pillow') #saving animation output in gif format

#plotting a graph of the steady state
import pandas as pd
import seaborn as sns

earth_cells = []
tree_cells = []
fire_cells = []

for cell_state in sim_steady:
    earth = (cell_state == 0).sum()
    earth_cells.append((earth / (rows * cols)) * 100)
    trees = (cell_state == 1).sum()
    tree_cells.append((trees / (rows * cols)) * 100)
    fire = (cell_state == 2).sum()
    fire_cells.append((fire / (rows * cols)) * 100)


earth_dataframe = pd.DataFrame({"Earth": earth_cells})
tree_dataframe = pd.DataFrame({"Trees": tree_cells})
fire_dataframe = pd.DataFrame({"Fire": fire_cells})

steady_state_plot_exearth = pd.concat([fire_dataframe, tree_dataframe], axis=1)

palette = {'Fire': '#FF4500','Trees': '#006400', 'Earth': 'brown'}
plt.figure(figsize=(10, 6))
sns.lineplot(data=steady_state_plot, palette=palette).set(xlabel='Time step', ylabel='Percentage of Cells', title='Percentage of Earth, Tree, and Fire Cells Over Time')
plt.legend(title='Cell Type', labels=['Fire', 'Trees', 'Earth'])
plt.savefig('Steady_state_investigation.png') #save the plot
#show the plot
plt.show() 


