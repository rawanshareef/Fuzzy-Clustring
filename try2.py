import random

# Step 0 - Initialize Data Set
X = [(1, 3), (2, 5), (4, 8), (7, 9)]

# Algorithm Parameters
num_points = len(X)
num_clusters = 2
m = 2.0
epsilon = 1e-4
iter_limit = 100

def print_matrix(mat, clusters_centers: bool = False):
    if clusters_centers:
        for i in range(num_clusters):
            print(f"Cluster {i} center: {mat[i]}")
        print("\n")
        return
    for i in range(num_points):
        print(f"Cluster {i} membership values: {mat[i]}")
    print("\n")

def init_mem_mat():
    mem_mat = []
    for _ in range(num_points):
        sum = 0
        tmp_lst = []
        for _ in range(num_clusters):
            val = random.random()
            tmp_lst.append(val)
            sum += val
        tmp_lst = list(map(lambda x: x/sum, tmp_lst))
        mem_mat.append(tmp_lst)
    print("\nInitial Membership Matrix Values:\n")
    print_matrix(mem_mat, False)
    print("\n")
    return mem_mat

def cluster_center(X, mem_mat, iteration):
    clusters = []
    for cluster_idx in range(num_clusters):
        cluster_x, cluster_y = 0, 0
        denominator = sum([mem_mat[i][cluster_idx]**m for i in range(num_points)])
        numerator_x = sum([X[i][0] * (mem_mat[i][cluster_idx]**m) for i in range(num_points)])
        numerator_y = sum([X[i][1] * (mem_mat[i][cluster_idx]**m) for i in range(num_points)])
        cluster_x = numerator_x / denominator
        cluster_y = numerator_y / denominator
        clusters.append((cluster_x, cluster_y))
    clusters.sort(key=lambda center: (center[0], center[1]))  # Sort by x, then y
    print_matrix(clusters, True)
    return clusters

def calculate_distances(X, centers):
    D = []
    for i in range(num_points):
        point_lst = []
        for center in range(num_clusters):
            distance = ((X[i][0] - centers[center][0])**2 + (X[i][1] - centers[center][1])**2)**0.5
            point_lst.append(distance)
        D.append(point_lst)
    return D

def update_mem_vals(mem_mat, D):
    pow = 1 / (m - 1)
    flag = False
    for i in range(num_points):
        for j in range(num_clusters):
            den = sum([(D[i][j] / (D[i][c] + 1e-9))**pow for c in range(num_clusters)])
            delta = abs(1 / den - mem_mat[i][j])
            flag |= delta > epsilon
            mem_mat[i][j] = 1 / den
    return flag


def fuzzy(X):
    # Step 1 - Initialize Membership Matrix
    mem_mat = init_mem_mat()
    #mem_mat = [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]]

    # Step 2-4 Loop
    iter_num = 1
    while iter_num <= iter_limit:
        # Step 2 - Calculate Cluster Centers
        centers = cluster_center(X, mem_mat, iter_num)

        # Step 3 - Calculate Distances
        D = calculate_distances(X, centers)

        # Step 4 - Update Membership Values
        if update_mem_vals(mem_mat, D) == False:
            print(f"\nConverged at iteration {iter_num}\n")
            break
        
        # Increment Loop Counter
        iter_num += 1
    
    return mem_mat, centers

def predict(X, centers):
    predictions = [0] * num_points
    for i in range(num_points):
        min_dist = float('inf')
        for j in range(num_clusters):
            dist = (X[i][0] - centers[j][0])**2 + (X[i][1] - centers[j][1])**2
            if dist < min_dist:
                min_dist = dist
                predictions[i] = j
    return predictions

def calc_accuracy(labels, class_labels):
    correct = 0
    for i in range(num_points):
        if labels[i] == class_labels[i]:
            correct += 1
    return correct / num_points

mat, centers = fuzzy(X)
print(f"Final membership matrix: {mat}\n")
print(f"Final clusters' centers: {centers}\n")

predictions = predict(X, centers)
for i in range(num_points):
    print(f"Point {X[i]} belongs to cluster {predictions[i]}")
accuracy = calc_accuracy(predictions, [0, 0, 1, 1])
print(f"\nClassification Accuracy = {accuracy*100}%\n")