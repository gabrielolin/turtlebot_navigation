import numpy as np
import heapq
import matplotlib.pyplot as plt
import cv2
import time

class Cell:
    def __init__(self):
        self.parent_i = 0 # parent cell row
        self.parent_j = 0 # parent cell column
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

MAX_ROW = 103
MAX_COL = 112

#check if cell is valid
def is_valid(row,col):
    return (row>=0) and (row<MAX_ROW) and (col >= 0) and (col<MAX_COL)

#check if cell is unblocked
def is_unblocked(grid,row,col):
    return grid[row][col]==0

#check if cell is destination
def is_destination(row,col,goal):
    return row == goal[0] and col == goal[1]

def read_png(filename):
    # Read the .png file into a color image using OpenCV
    color_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    return color_image

def modify_pixel(image, row, col, final):
    if final:
        image[row, col] = [0, 215, 255]  # Set the pixel to green in BGR format
    else:
        image[row, col] = [0, 255, 0]  # Set the pixel to green in BGR format

def trace_path(cell_details, goal,img):
    print("The Path is ")
    path = []
    row = goal[0]
    col = goal[1]
 
    # Trace the path from goal to start using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col
        modify_pixel(img,row,col,True)
    
    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from start to goal
    path.reverse()
    
    # Print the path
    for i in path:
        print("->", i, end=" ")
    print()
    return path

#calculate euclidean heuristic
def calculate_h(row,col,goal):
    return ((row-goal[0])**2+(col-goal[1])**2)**0.5

def a_star(grid,start,goal,img):
    #check if start and goal are valid
    if not is_valid(start[0], start[1]) or not is_valid(goal[0], goal[1]):
        print("Start or Goal is invalid")
        return
    
    #check if start and goal are unblocked
    if not is_unblocked(grid,start[0],start[1]) or not is_unblocked(grid,goal[0],goal[1]):
        print("Start or Goal is blocked")
    
    #check if already at goal
    if is_destination(start[0],start[1],goal):
        print("Alread at Goal")
        return
    
    #initialize closed list for visited cells
    closedList = [[False for _ in range(MAX_COL)] for _ in range(MAX_ROW)]
    #initialize cells
    cells = [[Cell() for _ in range(MAX_COL)] for _ in range(MAX_ROW)]

    #initialze start cell details
    i = start[0]
    j=start[1]
    cells[i][j].g=0
    cells[i][j].h=calculate_h(i,j,goal)
    cells[i][j].f=calculate_h(i,j,goal)
    cells[i][j].parent_i=i
    cells[i][j].parent_j=j

    #initialize open list (cells to be visited)
    openList = []
    heapq.heappush(openList,(cells[i][j].f,i,j))

    #flag for whether goal is found
    found_goal = False
    
    plt.ion()
    fig, ax = plt.subplots()
    img_plot = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('A* Algorithm Visualization')
    plt.axis('off')
    plt.show()

    #main loop
    while len(openList)>0:
        #pop cell with smallest f value
        p = heapq.heappop(openList)
        print(p)

        print(cells[p[1]][p[2]].f)

        if (p[1]!=start[0] or p[2]!=start[1]) and (p[1]!=goal[0] or p[1]!=goal[1]):
            modify_pixel(img, p[1], p[2],False)
            img_plot.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.draw()
            plt.pause(0.1)

        #mark cell as visited
        i = p[1]
        j= p[2]
        closedList[i][j]=True

        #check each direction
        directions = [(1, 1),(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1), (-1, -1)]
        for d in directions:
            new_i = i+d[0]
            new_j = j+d[1]

            if is_valid(new_i,new_j) and is_unblocked(grid,new_i,new_j) and not closedList[new_i][new_j]:
                if is_destination(new_i,new_j,goal):
                    cells[new_i][new_j].parent_i=i
                    cells[new_i][new_j].parent_j=j
                    print("Goal Found")
                    trace_path(cells,goal,img)
                    found_goal=True
                    return
                else:
                    #calc new f,g,h
                    g_new=cells[i][j].g+1.0
                    h_new = calculate_h(new_i,new_j,goal)
                    f_new = g_new + h_new
                    print(g_new)
                    # if cell not in open list or new f is smaller
                    if cells[new_i][new_j].f == float('inf') or cells[new_i][new_j].f > f_new:
                        heapq.heappush(openList,(f_new,new_i,new_j))
                        #print(f_new)
                        cells[new_i][new_j].f=f_new
                        cells[new_i][new_j].g=g_new
                        cells[new_i][new_j].h=h_new
                        cells[new_i][new_j].parent_i = i
                        cells[new_i][new_j].parent_j = j
    if not found_goal:
        print("Failed to find the destination cell")

    
    plt.ioff()
    plt.show()


def main():
    binaryMap = np.loadtxt('/home/gabrielolin/turtlebot3_ws/src/maps/binary_map.txt', dtype=int)
 

    print(binaryMap.shape)
    start_position = [16,27] 
    goal_position = [82,91]  
     # Read the .png file
    color_image = read_png('/home/gabrielolin/turtlebot3_ws/src/maps/grid.png')

    # Run A* algorithm and visualize it in real-time
    a_star(binaryMap,start_position,goal_position,color_image)
    cv2.imwrite('solved_path.png', color_image)
    

if __name__=='__main__':
    main()

