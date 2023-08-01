## Written by: Sreya Sarathy
## Attribution: Hugh Liu, Jayden Ye, Joshua Dietrich, and and Liuyu Chen
## Collaborated with Harshet Anand
import numpy as np
import math

# The width I had was 47 while the height I was given was 59.
width, height = 47, 59
center_idx = int((width-1)/2)
M = np.zeros([height*2+1, width*3+1]) # space

# example-maze is the example maze shown to you. You can copy and past
# the content in the example maze into a txt file and name it `example-maze.txt`
file = open('example-maze.txt', 'r')
data = []
for row in file:
    data.append(row.strip())

for h in range(height*2+1):
    for w in range(width*3+1):
        if data[h][w] == ' ':
            M[h,w] = 0 # 0 for ' '
        if data[h][w] == '+':
            M[h,w] = 1 # 1 for "+"
        if data[h][w] == '-':
            M[h,w] = 2 # 2 for '-'
        if data[h][w] == '|':
            M[h,w] = 3 # 3 for '|'
# print(M)
# print(M.shape)

# The provided code defines a Cell class and creates a 2D grid of Cell objects
# using a nested list comprehension.
# Each Cell object represents a cell in the grid with specific attributes and behavior.
class Cell:
    def _init_(self, i, j):
        self.i = i
        self.j = j
        self.succ = ''
        self.action = ''  # which action the parent takes to get this cell
cells = [[Cell(i,j) for j in range(width)] for i in range(height)]

succ_matrix = []
# for each row:
for i in range(1,len(data),2):
    curr_row = []
    # for each col:
    for j in range(1,len(data[0])-1,3):
        curr_cell = ''
        if data[i-1][j] == ' ':
            if i != 1: # prevent leaving the maze
                curr_cell += 'U'
        if data[i+1][j] == ' ':
            if i != len(data)-2: # prevent leaving the maze
                curr_cell += 'D'
        if data[i][j-1] == ' ':
            curr_cell += 'L'
        if data[i][j+2] == ' ':
            curr_cell += 'R'
        curr_row.append(curr_cell)
    succ_matrix.append(curr_row)

for i in range(height):
    for j in range(width):
        cells[i][j].succ = succ_matrix[i][j]

# The following is the Successor matrix
with open("Question2.txt", "w") as f:
    for cell_row in cells:
        f.write(",".join([cell_col.succ for cell_col in cell_row]) + "\n")


# The following lines of code  implements a breadth-first search (BFS) algorithm to
# explore a grid-based map. The BFS algorithm efficiently finds the shortest path from the starting point (0, center_idx)
# to the target (height - 1, center_idx) while keeping track of the actions taken to traverse the path.
visited = set()
# entrance:
s1 = {(0,center_idx)}
s2 = set()
while (height - 1, center_idx) not in visited:
    for a in s1:
        visited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in (s1 | s2 | visited):
            s2.add((i-1,j))
            cells[i-1][j].action = 'U'
        if 'D' in succ and (i+1,j) not in (s1 | s2 | visited):
            s2.add((i+1,j))
            cells[i+1][j].action = 'D'
        if 'L' in succ and (i,j-1) not in (s1 | s2 | visited):
            s2.add((i,j-1))
            cells[i][j-1].action = 'L'
        if 'R' in succ and (i,j+1) not in (s1 | s2 | visited):
            s2.add((i,j+1))
            cells[i][j+1].action = 'R'
    s1 = s2
    s2 = set()

# The following lines of code  represents a 2D grid of '1's and '0's that visually shows
# the visited cells during the exploration of a grid-based map. The code iterates through
# each cell in the grid and writes '1' to the file if the cell is in the visited set
# (which indicates it has been visited during the search),
# otherwise, it writes '0'. Each row in the file represents a row in the grid, and the cells are separated by commas.
with open("Question5.txt", "w") as f:
    for h in range(height):
        for w in range(width):
            f.write("1" if (h, w) in visited else "0")
            if w != width - 1:
                f.write(",")
        f.write("\n")

cur = (height - 1, center_idx)
s = ''
seq = []
while cur != (0, center_idx):
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    s += t
    if t == 'U': cur = (i+1, j)
    if t == 'D': cur = (i-1, j)
    if t == 'L': cur = (i, j+1)
    if t == 'R': cur = (i, j-1)
action = s[::-1] # reverse
# action sequence
with open("Question3.txt", "w") as f:
    f.write(action + "\n")


seq.append((0, center_idx))
seq = seq[::-1]

# The following lines of code update the M array to include the special points
# represented by the integer value 4.
# It iterates through a sequence of (a, b) pairs and modifies specific positions
# in the grid to mark the special points.
for (a,b) in seq:
    M[2*a+1, 3*b+1] = 4
    M[2*a+1, 3*b+2] = 4
    if (a+1,b) in seq and M[2*a+2, 3*b+1] != 2:
        M[2*a+2, 3*b+1] = 4
        M[2*a+2, 3*b+2] = 4

    if (a,b-1) in seq and M[2*a+1, 3*b] != 1 and M[2*a+1, 3*b] != 3:
        M[2*a+1, 3*b] = 4

M[0,3*center_idx+1] = 4
M[0,3*center_idx+2] = 4


M[2*height,3*center_idx+1] = 4
M[2*height,3*center_idx+2] = 4

# The following lines of code are for the maze solution in the project.
# This code snippet writes a representation of a grid-based map stored in the
# M array to a file named "Question4.txt". The grid is represented using different characters to
# display various elements of the map.
with open("Question4.txt", "w") as f:
    for h in range(height*2+1):
        for w in range(width*3+1):
            if M[h,w]==0:
                f.write(' ')
            elif M[h,w]==1:
                f.write('+')
            elif M[h,w]==2:
                f.write('-')
            elif M[h,w]==3:
                f.write('|')
            elif M[h,w]==4:
                f.write('@')
        f.write('\n')


# The following lines of code implements a breadth-first search (BFS) algorithm to
# explore a grid-based map until the target point at coordinates (height - 1, center_idx) is reached.
# The algorithm maintains two sets s1 and s2 to keep track of points to be explored in the current
# and next levels of the BFS, respectively.
visited = set()
s1 = [(0, center_idx)]
s2 = set()

# The following algorithm explores the grid by visiting the successors of the points
# in s1 in a depth-first manner. It adds each point to the visited set, checks its valid successors,
# and adds them to s2 if they meet the conditions.
# The algorithm backtracks by reversing the order of points in s1 after processing each level.
while (height-1, center_idx) not in visited:
    for a in s1:
        visited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i - 1, j) not in (s2 | visited) and (i - 1, j) not in s1:
            s2.add((i - 1, j))
            cells[i - 1][j].action = 'U'
        if 'D' in succ and (i + 1, j) not in (s2 | visited) and (i + 1, j) not in s1:
            s2.add((i + 1, j))
            cells[i + 1][j].action = 'D'
        if 'L' in succ and (i, j - 1) not in (s2 | visited) and (i, j - 1) not in s1:
            s2.add((i, j - 1))
            cells[i][j - 1].action = 'L'
        if 'R' in succ and (i, j + 1) not in (s2 | visited) and (i, j + 1) not in s1:
            s2.add((i, j + 1))
            cells[i][j + 1].action = 'R'
    for b in s2:
        s1.append(b)
    s1.reverse()
    s2 = set()

# The following lines of code are used to generate DFS which is used to generate the output for Q6
# The code iterates through each row and column of the grid. If the point (h, w)
# is in the visited set, it writes '1' to the file, indicating that the point has been visited.
# Otherwise, it writes '0' to indicate that the point has not been visited yet.
# Distinct points in each row are separated by commas, and each row is written on a new line.
with open("Question6.txt", "w") as f:
    for h in range(height):
        for w in range(width):
            f.write("1" if (h, w) in visited else "0")
            if w != width - 1:
                f.write(",")
        f.write("\n")


# For the Manhattan distance, the formula is:
# abs(i - (height - 1)) + abs(j - center_idx),
# which calculates the vertical and horizontal distances
# from each point (i, j) to the goal point (height - 1, center_idx).
# For the Euclidean distance, the formula is: math.sqrt((i - (height - 1)) ** 2 + (j - center_idx) ** 2),
# which calculates the direct distance (hypotenuse) from each point (i, j) to the goal point (height - 1, center_idx)
# using the Pythagorean theorem.
man = {(i,j): abs(i-(height - 1)) + abs(j-center_idx) for j in range(width) for i in range(height)}
euc = {(i,j): math.sqrt((i-(height-1))*2 + (j-center_idx)*2 ) for j in range(width) for i in range(height)}

# The following lines of code iterates through each row and column of the grid and writes
# the corresponding Manhattan distance to the file. Distances for each square are separated by commas,
# and each row is written on a new line. After writing all the distances to the file,
# it will be saved as "Question7.txt" in the current working directory.
with open("Question7.txt", "w") as f:
    for h in range(height):
        for w in range(width):
            f.write(str(man[(h, w)]))
            if w != width - 1:
                f.write(",")
        f.write("\n")

# The following function defines a_star_search:
# that implements the A* algorithm for grid-based pathfinding.
# It initializes the cost dictionary g with initial values of infinity for all grid points,
# except for the starting point (0, center_idx), which is set to 0. The algorithm maintains a priority
# queue with the starting point, and a set visited to keep track of the visited nodes.
def a_star_search(height, width, dist_method, man, euc):
    g = {(i,j): float('inf') for j in range(width) for i in range(height)}
    g[(0, center_idx)] = 0

    queue = [(0,center_idx)]
    visited = set()

# The algorithm maintains a priority queue of points to be explored based on the given
# distance heuristic (Manhattan or Euclidean). The code efficiently expands the search space by
# updating the minimum cost g for each explored point and considering the valid successor cells.
# The function returns the set of visited points during the A* algorithm execution.
    while queue and (height - 1,center_idx) not in visited:
        if dist_method == 'manhattan':
            queue.sort(key=lambda x: g[x] + man[x])
        elif dist_method == 'euclidean':
            queue.sort(key=lambda x: g[x] + euc[x])
        else:
            print('distance method should be either mahattan or euclidean!')
        point = queue.pop(0)

# The following lines of code are a part of the A* Algorithm's main loop to explore the grid - based map and update
# the visited nodes based on the successors of the current point. It follows the A* algorithm's logic
# to consider the available directions (Up, Down, Left, Right) from the current point (i, j) and update their
# costs if they have not been visited yet. It also updates the cost g for each point if a more efficient path is found.
        if point not in visited:
            visited.add(point)
            i, j = point[0], point[1]
            succ = cells[i][j].succ
            if 'U' in succ and (i-1,j) not in visited:
                if (i-1,j) not in queue: queue += [(i-1,j)]
                g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
            if 'D' in succ and (i+1,j) not in visited:
                if (i+1,j) not in queue: queue += [(i+1,j)]
                g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
            if 'L' in succ and (i,j-1) not in visited:
                if (i,j-1) not in queue: queue += [(i,j-1)]
                g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
            if 'R' in succ and (i,j+1) not in visited:
                if (i,j+1) not in queue: queue += [(i,j+1)]
                g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)
    return visited

# list of squares searched by A* with Manhattan distance to the goal as the heuristic
a_star_man_visited = a_star_search(height, width, 'manhattan', man, euc)

# list of squares searched by A* with Euclidean distance to the goal as the heuristic
a_star_euclidean_visited = a_star_search(height, width, 'euclidean', man, euc)

# The following lines of code are used to generate the answer for Question 8 of the project
# This code segment writes a 2D grid of '1's and '0's to "Question8.txt,"
# indicating visited nodes in A* algorithm using Manhattan distance heuristic.
with open("Question8.txt", "w") as f:
    for h in range(height):
        for w in range(width):
            f.write("1" if (h, w) in a_star_man_visited else "0")
            if w != width - 1:
                f.write(",")
        f.write("\n")

# The following lines of code are used to generate the answer for Question 9 of the project.
# Moreover, this code snippet writes a 2D grid of '1's and '0's to a file,
# representing visited nodes in A* algorithm with Euclidean distance heuristic.
with open("Question9.txt", "w") as f:
    for h in range(height):
        for w in range(width):
            f.write("1" if (h, w) in a_star_euclidean_visited else "0")
            if w != width - 1:
                f.write(",")
        f.write("\n")