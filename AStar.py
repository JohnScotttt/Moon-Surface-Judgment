from Jtools import *

def heuristic(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def astar(start, end, image):
    height, width = image.shape[:2]
    visited = np.zeros((height, width), dtype=bool)
    parent = np.zeros((height, width, 2), dtype=int)
    g_score = np.full((height, width), np.inf)
    f_score = np.full((height, width), np.inf)

    g_score[start] = 0
    f_score[start] = heuristic(start, end)

    queue = PriorityQueue()
    queue.put((f_score[start], start))

    while not queue.empty():
        current = queue.get()[1]
        if current == end:
            break

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor[0] < 0 or neighbor[0] >= height or neighbor[1] < 0 or neighbor[1] >= width:
                continue
            if visited[neighbor]:
                continue
            if np.array_equal(image[neighbor], [128, 0, 1]):
                continue

            new_g_score = g_score[current] + 1
            if new_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = new_g_score
                f_score[neighbor] = new_g_score + heuristic(neighbor, end)
                queue.put((f_score[neighbor], neighbor))
                visited[neighbor] = True

    path = []
    current = end
    while current != start:
        path.append(current)
        current = tuple(parent[current])
    path.append(start)
    path.reverse()

    return path

# Read the image
image = cv2.imread('dom.png')

# Define the start and end points
start = (346, 648)
end = (1, 662)

# Apply A* algorithm to find the path
path = astar(start, end, image)

# Draw the path on the image
for point in path:
    image[point] = [0, 255, 0]  # Green color for the path

# Save the image with the path drawn on it
cv2.imwrite('AStar.png', image)
