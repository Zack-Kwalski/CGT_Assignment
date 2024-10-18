import function as func

user_input = input("Enter a list of degrees separated by spaces: ")
degree_sequence = list(map(int, user_input.split()))
print("Your list of degree sequence:", degree_sequence)

# Create the graph from the degree sequence
graph = func.create_graph_from_deg_sequence(degree_sequence)
graph_dict = func.deg_seq_to_dict(degree_sequence)
weighted_graph = func.assign_random_weights(graph_dict)

def display_menu():
    print("\nChoose an operation:")
    print("1. Plot Graph")
    print("2. Check if Graph is Eulerian")
    print("3. Find Eulerian Path")
    print("4. Run Dijkstra Algorithm")
    print("5. Minimum Spanning Tree")
    print("6. edge-vertex-K-Connected")
    print("7. Draw Weighted Graph")
    print("8. Show Degree Sequence as Dictionary")
    print("9. Show weighted graph")
    print("0. Exit")

def main():
    while True:
        display_menu()
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid number.")
            continue
        
        if choice == 1:
            if graph:
                func.plot_graph(graph)
            else:
                print("Invalid graph!")
        elif choice == 2:
            if graph:
                func.is_eulerian(graph)
            else:
                print("Invalid graph!")
        elif choice == 3:
            print("Euler Path: ", func.find_eulerian_path(graph_dict))
        elif choice == 4:
            start_node = int(input("Enter the starting node for Dijkstra's algorithm: "))
            dijkstra_result = func.dijkstra(weighted_graph, start_node)
            print(dijkstra_result)
        elif choice == 5:
            prim_mst = func.prim(weighted_graph, 0)
            print("Prim's MST:", dict(prim_mst))
            func.draw_weighted_graph(prim_mst)
        elif choice == 6:
            # Find vertex connectivity
            print(graph_dict)
            vertex_conn_value, vertices_removed = func.vertex_connectivity(graph_dict)
            print(f"Vertex Connectivity: {vertex_conn_value}")
            print(f"Vertices to remove to disconnect the graph: {vertices_removed}")

            # Find edge connectivity
            edge_conn_value, edges_removed = func.edge_connectivity(graph_dict)
            print(f"Edge Connectivity: {edge_conn_value}")
            print(f"Edges to remove to disconnect the graph: {edges_removed}")

            # K-connected graph is based on vertex connectivity
            K_connected = vertex_conn_value
            print(f"The graph is {K_connected}-connected.")
        elif choice == 7:
            func.draw_weighted_graph(weighted_graph)
        elif choice == 8:
            print(graph_dict)
        elif choice == 9:
            print(weighted_graph)
        elif choice == 0:
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please choose a valid option.")

# Run the main function
if __name__ == "__main__":
    main()