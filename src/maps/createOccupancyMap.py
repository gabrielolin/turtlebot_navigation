import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

def max_pooling(grid, factor):
    # Calculate the shape of the downsampled grid
    new_shape = (grid.shape[0] // factor, grid.shape[1] // factor)
    # Initialize the downsampled grid
    downsampled_grid = np.zeros(new_shape, dtype=np.uint8)
    
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # Define the block
            block = grid[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            # Compute the maximum value of the block and assign it to the downsampled grid
            downsampled_grid[i, j] = np.max(block).astype(np.uint8)
    
    return downsampled_grid


def inflate_obstacles(grid, inflation_radius):
    inflated_grid = np.copy(grid)
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            if grid[row, col] == 1:  # obstacle
                for dy in range(-inflation_radius, inflation_radius + 1):
                    for dx in range(-inflation_radius, inflation_radius + 1):
                        new_row, new_col = row + dy, col + dx
                        if 0 <= new_row < height and 0 <= new_col < width:
                            if grid[new_row][new_col]==0:
                                inflated_grid[new_row, new_col] = 2  # Mark as inflated obstacle
    return inflated_grid

def process_and_save_as_png(grid, filename, start, goal, inflation_radius):
    # Inflate obstacles
    inflated_grid = inflate_obstacles(grid, inflation_radius)
    
    # Invert the values: 0 -> 255 (white), 1 -> 0 (black), 2 -> 127 (grey)
    inverted_grid = np.where(inflated_grid == 0, 255, np.where(inflated_grid == 1, 0, 127))
    
    # Ensure the grid is of type uint8
    inverted_grid = inverted_grid.astype(np.uint8)
    
    # Convert to color image
    color_grid = cv2.cvtColor(inverted_grid, cv2.COLOR_GRAY2BGR)
    
    # Mark start position in red
    color_grid[start[0], start[1]] = [0, 0, 255]  # Red in BGR
    
    # Mark goal position in blue
    color_grid[goal[0], goal[1]] = [255, 0, 0]  # Blue in BGR
    
    # Save the processed grid as a .png file
    cv2.imwrite(filename, color_grid)
	# Return the processed grid for further use
    return color_grid


#Load the Map
map_image = cv2.imread('frontier_map.pgm', cv2.IMREAD_GRAYSCALE)


#Threshold the image
_, binary_map = cv2.threshold(map_image, 200, 1, cv2.THRESH_BINARY_INV)

binary_grid = binary_map.astype(np.uint8)
factor = 2
downsample_array = max_pooling(binary_grid,factor)



output_filename = 'grid.png'
start_position = (16, 27) 
goal_position = (82, 91)  


# Process and save the grid, and return it for plotting
processed_img = process_and_save_as_png(binary_grid, output_filename, start_position, goal_position,3)
inflated_grid = inflate_obstacles(binary_grid,3)
np.savetxt('binary_map.txt', inflated_grid, fmt='%d')

print(binary_grid.shape)

# Plot the processed grid using Matplotlib
plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
plt.title('Occupancy Grid with Start and Goal')
plt.show()











