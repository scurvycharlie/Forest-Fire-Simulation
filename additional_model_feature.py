##For the additional model feature, the simulation was adapted to simulate a more species diverse ecology in terms of vegitation. The idea of the simulation is that the fire will burn slowly in some tree species and the fire will also spread slower, but the trees will grow back slower too. Other species will burn quickly and spread quickly as well, while also populating empty spaces faster than slow growing and burning species. 

#Additionally the effects of introducing eucalyptus into the environment were investigated, eucalyptus is fire resistant, produces highly flammable oil which can explode and spread rapidly. To simulate the spitting the eucalyptus sets any trees within 2 blocks of it ablaze, also it takes several timesteps to return to a 0. 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import random

def retrieve_neighbouring_cells(grid, x, y): 
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

def retrieve_distant_cells(grid, x, y):
    """
    This function performs the exact same task as the retrieve cells function, but for cells 2 away from the target
    """
    rows, cols = grid.shape
    distant_cells = []
    
    if x > 1: #if x coordinate isn't the 0 or 1 index
        distant_cells.append(grid[x - 2, y]) #collect the cell upwards of it
    if x < rows - 2: #if the cell is not in the last row
        distant_cells.append(grid[x + 2, y]) #collect the cell below it
    if y > 1: #if the column is not the 0 or 1 index
        distant_cells.append(grid[x, y - 2]) #collect the cell to the left (-2)
    if y < cols - 2: #if the col is not the last col
        distant_cells.append(grid[x, y + 2]) #collect the cell to the right (+2)
    
    return distant_cells

def fire_check(neighbouring_cells):
    for n in neighbouring_cells: 
        if 5 <= n <= 8:
            return True 
    return False

def eucalyptus_fire_check(distant_cells):
    """
    Function checks for the presence of 5 in the (+-) 2 cells. The only value whcih becomes 5 is 4, so this function simulates a neaby burning Eucalyptus tree
    """
    for n in distant_cells:
        if n == 5:
            return True
    return False

def update_state(grid, probability_of_new_growth, probability_of_lightning_strike):
    grid_new = np.copy(grid)
    rows, cols = grid.shape
    
    plants = [1, 2, 3, 4]  # 1: Pine, 2: Oak, 3: Brambles, 4: Eucalyptus
    #probabilities = [0.14, 0.08, 0.23, 0.17]  # Probabilities corresponding to each plant, for probability_of_regrowth to be effectively applied, user must ensure the sum of these four values = 1
    probabilities = [0.3, 0.1, 0.3, 0.3]
    
    for x in range(rows):
        for y in range(cols):
            target_cell = grid[x, y]
            neighbouring_cells = retrieve_neighbouring_cells(grid, x, y) #determinging neighbours 
            distant_cells = retrieve_distant_cells(grid, x, y) #determing distant cells 
            FIRE = fire_check(neighbouring_cells) #checking for fire in neighbours 
            EUCALYPTUS_FIRE = eucalyptus_fire_check(distant_cells) #checking for burning eucalyptus in far cells
            
            if target_cell == 1 and FIRE or EUCALYPTUS_FIRE:
                grid_new[x, y] = 7
                
            elif target_cell == 2 and FIRE or EUCALYPTUS_FIRE:
                grid_new[x, y] = 6

            elif target_cell == 3 and FIRE or EUCALYPTUS_FIRE:
                grid_new[x, y] = 8
                
            elif target_cell == 4 and FIRE or EUCALYPTUS_FIRE:
                grid_new[x, y] = 5

            # Transition from burning to burnt, this enables some trees to burn for multiple time steps. 
            elif target_cell == 5:
                grid_new[x, y] = 6 #eucalyptus fire transitions to 6
            elif target_cell == 6:
                grid_new[x, y] = 7 #fire level six goes to seven
            elif target_cell == 7:
                grid_new[x, y] = 8 #fire level seven goes to eight
            elif target_cell == 8:
                grid_new[x, y] = 0

            elif 1 <= target_cell <= 4 and np.random.rand() < probability_of_lightning_strike:
                grid_new[x, y] = 8 #updates any tree hit by lightining to fire level 8. 
                    
            elif target_cell == 0 and np.random.rand() < probability_of_new_growth:
                grid_new[x, y] = np.random.choice([0] + plants, p=[1-sum(probabilities)] + probabilities) #this adds the 0 to the plants array and applies the probability of each plant growing. 
            
    return grid_new

def run_simulation(forest, num_steps):
    forest_state = [forest]  # state is a list which stores the state of the forest for each step
    for step in range(num_steps): 
        forest = update_state(forest, probability_of_new_growth, probability_of_lightning_strike)
        forest_state.append(forest) #for loop runs the update_state on the returned forest for each step defined by num_steps
    return forest_state #return the stored simulated states

rows = 100
cols = 100
num_steps = 1000
probability_of_new_growth = 0.002
probability_of_lightning_strike = 0.00001


sample_array = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
# Corrected color map
cmap = ListedColormap([
    '#653700',  # Brown 
    '#006400',  # Pine tree (Dark Green)
    '#228B22',  # Oak tree (Medium Green)
    '#90EE90',  # Brambles (or other applicable fast growing shrub) (Light Green)
    '#3CB371',  # Eucalyptus (Greeny blue)
    '#FFFFE0',  # Fire level 1 (Light Yellow)
    '#FFD700',  # Fire level 2 (Gold)
    '#FFA500',  # Fire level 3 (Orange)
    '#FF4500',  # Fire level 4 (Orange-Red)
])

forest = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], size=(rows, cols)) #setting up the forest as a grid, all values have to be included to correctly assign the colours
print(np.unique(forest)) #checking for unique values which would disrupt the simulation


plt.imshow(sample_array, cmap=cmap) #checking the colour palette is correct

sim2 = run_simulation(forest, num_steps) #running the simulation

#assert not np.any(sim2[0] >= 5), "Fire states found in the first step of the simulation"
fig, ax = plt.subplots()
plot = ax.imshow(sim2[0], cmap = cmap) #checking the intial grid. 

def animate(i):
    plot.set_data(sim2[i])
    return [plot]

anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False) #run the animation of the simulation

anim.save('eucalyptus_forest_fire_simulation.gif', writer='pillow') #saving animation output in gif format
anim.save('eucalyptus_forest_fire_simulation.mp4', writer='ffmpeg') #saving animation output in mp4 format


import pandas as pd
import seaborn as sns

earth_cells = [] 
pine_cells = []
oak_cells = []
bramble_cells = []
eucalyptus_cells = []
fire_cells = []

for cell_state in sim2:
    earth = (cell_state == 0).sum()
    earth_cells.append((earth / (rows * cols)) * 100)
    pine_trees = (cell_state == 1).sum()
    pine_cells.append((pine_trees / (rows * cols)) * 100)
    oak_trees = (cell_state == 2).sum()
    oak_cells.append((oak_trees / (rows * cols)) * 100)
    brambles = (cell_state == 3).sum()
    bramble_cells.append((brambles / (rows * cols)) * 100)
    eucalyptus_trees = (cell_state == 4).sum() 
    eucalyptus_cells.append((eucalyptus_trees / (rows * cols)) * 100)
    fire = ((cell_state >= 5) & (cell_state <= 8)).sum()
    fire_cells.append((fire / (rows * cols)) * 100)


earth_dataframe = pd.DataFrame({"Earth": earth_cells}) #set the earth dataframe
pine_dataframe = pd.DataFrame({"Pine": pine_cells}) #set the pine dataframe
oak_dataframe = pd.DataFrame({"Oak": oak_cells}) #set the oak dataframe
bramble_dataframe = pd.DataFrame({"Bramble": bramble_cells}) #set the bramble dataframe
eucalyptus_dataframe = pd.DataFrame({"Eucalyptus": eucalyptus_cells}) #set the eucalyptus dataframe
fire_dataframe = pd.DataFrame({"Fire": fire_cells}) #set the fire dataframe

eucalyptus_sim2 = pd.concat([fire_dataframe, pine_dataframe, oak_dataframe, bramble_dataframe, eucalyptus_dataframe], axis=1) #concatenate the results into a single dataframe for time plot analysis 

palette = {'Fire': '#FF4500','Pine': '#006400','Oak': '#228B22','Bramble': '#90EE90','Eucalyptus': '#3CB371'} #setting the palette colours to the hues used in the grapgh, fire is just orange, Seaborn instruction manual

plt.figure(figsize=(10, 6)) #setting fig size 
sns.lineplot(data=eucalyptus_sim2, palette=palette).set(xlabel='Time step', ylabel='Percentage of Cells', title='Percentage of Tree Species Coverage Over Time') #setting up the lineplot with corrected colour palette
plt.legend(title='Cell Type', labels=['Fire', 'Pine', 'Oak', 'Bramble', 'Eucalyptus']) #setting the legend
plt.ylim(0,17) #initialisation causes odd values, limit the ylim for easier viewing
plt.xlim(10, 1000) #remove initial simulation chaos from plot
plt.savefig('Steady_state_investigation_eucalyptus.png') #save the plot
plt.show() #show the timestep evolution of the simulation