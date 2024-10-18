import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import random
random.seed(69)

def havel_hakimi(seq):
    # if seq is empty or only contains zeros,
    # degree sequence is valid
    if len(seq) < 1 or all(deg == 0 for deg in seq):
        print("Finished! Graph IS constructable with Havel Hakimi algorithm.")
        return True

    print(seq, end="")
    seq.sort(reverse=True)  # Sort in non-increasing (descending) order
    print(" --sort--> ", end="")
    print(seq)

    first = seq[0]  # The largest element now becomes the first one
    if first > len(seq) - 1:
        print("Failed! Graph IS NOT constructable with Havel Hakimi algorithm.")
        return False

    print(seq, end="")

    # remove the largest element (which is now at index 0)
    seq.pop(0)

    # iterate seq forwards and reduce elements
    for num in range(first):
        if seq[num] > 0:
            seq[num] -= 1
        else:
            print("\nFailed! Graph is not constructable with Havel Hakimi algorithm")
            return False

    print(" --alg-->", end="")
    print(seq)

    # recursive call
    return havel_hakimi(seq)
    #returns True or False

def is_valid_deg_sequence(deg_seq):
    # Check if the degree sequence is valid using the Erdős-Gallai theorem
    return havel_hakimi(deg_seq.copy())

def create_graph_from_deg_sequence(deg_seq):
    # Check if the degree sequence is valid
    if not is_valid_deg_sequence(deg_seq):
        return None
    
    # Create the graph using the Havel-Hakimi algorithm
    G = nx.havel_hakimi_graph(deg_seq)
        
    return G

def plot_graph(G):
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.show()

# Function for Depth First Search (DFS)
def dfs(graph, vertex, visited):
    visited.add(vertex)
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Check if the graph is connected
# def is_connected(graph):
#     start_node = next((node for node in graph if len(graph[node]) > 0), None)
#     if start_node is None:
#         return True  # Trivially connected if no edges

#     visited = set()

#     dfs(graph, start_node, visited)

#     for node in graph:
#         if len(graph[node]) > 0 and node not in visited:
#             return False

#     return True

def is_connected(graph):
    # Step 1: Find the first node with at least one connection (non-isolated node)
    start_node = next((node for node in graph if graph[node]), None)
    
    if start_node is None:
        # If no edges exist (all nodes isolated), the graph is trivially connected
        return True  

    visited = set()

    # Depth First Search (DFS) helper function
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)

    # Step 2: Start DFS from the first available node
    dfs(start_node)

    # Step 3: Verify that all nodes with edges have been visited
    for node in graph:
        if graph[node] and node not in visited:  # Only check nodes with edges
            return False

    return True

def deg_seq_to_dict(degree_sequence):
    # Check if the sum of degrees is even (necessary condition for a graphical sequence)
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("The degree sequence is not graphical (sum of degrees must be even).")
    
    # Initialize an empty adjacency list for the graph
    graph = defaultdict(list)
    
    # Assign node labels from 0 to len(degree_sequence) - 1
    nodes = list(range(len(degree_sequence)))
    
    # Work on a sorted version of the degree sequence
    degree_sequence = sorted(zip(degree_sequence, nodes), reverse=True)  # (degree, node)
    
    # While there are remaining degrees
    while degree_sequence:
        # Sort by the remaining degrees in descending order
        degree_sequence.sort(reverse=True, key=lambda x: x[0])

        # Get the node with the highest degree
        degree, node = degree_sequence.pop(0)
        
        if degree > len(degree_sequence):
            raise ValueError("The degree sequence is not graphical (too few nodes left to satisfy the degree).")
        
        # Connect this node to the next highest-degree nodes
        for i in range(degree):
            graph[node].append(degree_sequence[i][1])
            graph[degree_sequence[i][1]].append(node)
            # Reduce their degree by 1
            degree_sequence[i] = (degree_sequence[i][0] - 1, degree_sequence[i][1])
        
        # Remove nodes whose degree is now 0
        degree_sequence = [(d, n) for d, n in degree_sequence if d > 0]
    
    return dict(graph)


def is_valid_next_edge(graph, u, v):
    # Check if the edge u-v is valid to traverse
    if len(graph[u]) == 1:
        return True
    else:
        # Check if it's a bridge by using DFS (depth-first search) to count components
        original_component_count = count_components(graph, u)
        
        # Temporarily remove the edge
        graph[u].remove(v)
        graph[v].remove(u)
        
        # Recalculate components after removing the edge
        new_component_count = count_components(graph, u)
        
        # Restore the edge back
        graph[u].append(v)
        graph[v].append(u)
        
        # Check if it was a bridge
        return original_component_count == new_component_count


def count_components(graph, u):
    visited = set()
    
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
    
    # Start DFS from the given node u
    dfs(u)
    return len(visited)

# Find eulerian Path
def find_eulerian_path(graph):
    # Copy the graph to avoid modifying the original graph
    graph_copy = {node: neighbors[:] for node, neighbors in graph.items()}
    
    # Find a starting point (for Eulerian or semi-Eulerian graph)
    start_node = None
    odd_degree_nodes = [node for node in graph_copy if len(graph_copy[node]) % 2 == 1]
    
    if len(odd_degree_nodes) == 0:
        # Eulerian circuit: start from any node
        start_node = list(graph_copy.keys())[0]
    elif len(odd_degree_nodes) == 2:
        # Semi-Eulerian path: start from one of the odd-degree nodes
        start_node = odd_degree_nodes[0]
    else:
        return "The graph is not Eulerian or semi-Eulerian."
    
    # Fleury's algorithm to find the Eulerian path
    path = []
    def fleury(u):
        while graph_copy[u]:
            v = graph_copy[u][0]
            if is_valid_next_edge(graph_copy, u, v):
                # Remove the edge from the graph
                graph_copy[u].remove(v)
                graph_copy[v].remove(u)
                
                # Traverse this edge
                path.append((u, v))
                fleury(v)
    
    # Start traversal from the start node
    fleury(start_node)
    return path

# Check if the graph is Eulerian
def is_eulerian(graph):
    odd_degree_nodes = [node for node in graph if len(graph[node]) % 2 != 0]

    if not is_connected(graph):
        print("The graph is not connected, so it cannot be Eulerian.")
        return False
    
    if len(odd_degree_nodes) == 0:
        print("The graph has an Eulerian circuit.")
        return True
    elif len(odd_degree_nodes) == 2:
        print(f"The graph has an Eulerian path. The odd degree vertices are: {odd_degree_nodes}")
        return True
    else:
        print(f"The graph is not Eulerian. The vertices with odd degrees are: {odd_degree_nodes}")
        return False

# Function to assign random weights to edges
def assign_random_weights(graph):
    weighted_graph = {}
    
    for node, neighbors in graph.items():
        weighted_graph[node] = {}
        for neighbor in neighbors:
            if neighbor not in weighted_graph:  # Initialize neighbor's dictionary
                weighted_graph[neighbor] = {}
                
            # Assign a random weight to the edge (node, neighbor) and (neighbor, node)
            if neighbor not in weighted_graph[node]:  # Only assign if not already assigned
                weight = random.randint(1, 10)  # You can change the range as needed
                weighted_graph[node][neighbor] = weight
                weighted_graph[neighbor][node] = weight  # Same weight for reverse direction

    return weighted_graph

# Dijkstra's algorithm implementation
def dijkstra(graph, start):
    # Step 1: Initialization
    # Set initial distances to infinity, and the distance to the start node as 0
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Priority queue to get the node with the smallest distance
    priority_queue = [(0, start)]
    
    while priority_queue:
        # Step 2: Visit node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Step 3: Skip processing if we already found a shorter path
        if current_distance > distances[current_node]:
            continue
        
        # Step 4: Calculate tentative distances to neighbors
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # Step 5: Update if a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Function to draw weighted graph
def draw_weighted_graph(graph):
    # Transforming the input to match expected format
    weighted_graph = {u: {v: {'weight': w} for v, w in edges.items()} for u, edges in graph.items()}
    G = nx.from_dict_of_dicts(weighted_graph)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=16, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color='red')
    
    plt.title("Graph Visualization")
    plt.show()

# Function to remove vertices from the graph
def remove_vertices(graph, vertices_to_remove):
    new_graph = {}
    for vertex, neighbors in graph.items():
        if vertex not in vertices_to_remove:
            new_graph[vertex] = [n for n in neighbors if n not in vertices_to_remove]
    return new_graph

# Function to remove edges from the graph
def remove_edges(graph, edges_to_remove):
    new_graph = {}
    for vertex, neighbors in graph.items():
        new_neighbors = []
        for neighbor in neighbors:
            if (vertex, neighbor) not in edges_to_remove and (neighbor, vertex) not in edges_to_remove:
                new_neighbors.append(neighbor)
        new_graph[vertex] = new_neighbors
    return new_graph

# Function to calculate vertex connectivity
def vertex_connectivity(graph):
    print(graph)
    vertices = list(graph.keys())
    print(vertices)
    n = len(vertices)
    
    for k in range(1, n):
        for i in range(len(vertices)):
            print("First loop")
            for j in range(i + 1, len(vertices)):
                print("secind loop")
                for l in range(j + 1, len(vertices)):
                    print("third loop")
                    vertices_to_remove = [vertices[i], vertices[j], vertices[l]]
                    new_graph = remove_vertices(graph, vertices_to_remove)
                    print(new_graph)
                    if not is_connected(new_graph):
                        print("innermost loop")
                        return len(vertices_to_remove), vertices_to_remove

    return n - 1, []

# Function to calculate edge connectivity
def edge_connectivity(graph):
    edges = []
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            if (vertex, neighbor) not in edges and (neighbor, vertex) not in edges:
                edges.append((vertex, neighbor))

    for k in range(1, len(edges) + 1):
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                for l in range(j + 1, len(edges)):
                    edges_to_remove = [edges[i], edges[j], edges[l]]
                    new_graph = remove_edges(graph, edges_to_remove)
                    if not is_connected(new_graph):
                        return len(edges_to_remove), edges_to_remove

    return len(edges), []

# Prim's Algorithm
def prim(graph, start):
    mst = defaultdict(dict)
    visited = set([start])
    edges = [(cost, start, to) for to, cost in graph[start].items()]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst[frm][to] = cost  # Store the edge in the dict
            for next_to, next_cost in graph[to].items():
                if next_to not in visited:
                    heapq.heappush(edges, (next_cost, to, next_to))

    return mst