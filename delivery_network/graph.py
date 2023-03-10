class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.list_of_neighbours = []


    def dfs(self, node, node_visited):
        """
        This function is a depth-first-search algorithm.
        Parameters : 
        node : the starting node of the DFS
        node_visited : a dict representing if a gievn node (key of the dictionary) has been visited or not (value of the dictionary is a Boolean)
        """
  
        component = [node] #Initialization

        for neighbour in self.graph[node]:
            neighbour = neighbour[0]

            if not node_visited[neighbour]:
                node_visited[neighbour] = True #Update the status as this node has been visited
                component += self.dfs(neighbour, node_visited) #Add it to the component list

        return component

    def connected_components(self):
        """
        Builds a list of connected nodes, grouped in lists, and returns it.
        This function relies on the depth-first search alogrithm, implemented in dfs().
        """
        list_component = []
        node_visited = {node:False for node in self.nodes}

        for node in self.nodes:
            if not node_visited[node]:
                list_component.append(self.dfs(node, node_visited))

        return list_component


    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        self.nb_edges += 1
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """


    def bfs(self, start, power=-1):
        visited = set()  # Initializes to an empty set
        queue = [start]  # Initializes the queue with the starting node

        while queue:
            node = queue.pop(0)  # Dequeues the node at the front of the queue

            if node not in visited:
                visited.add(node)  # Marks the node as visited

                for neighbor, weight in self.graph[node]:
                    if weight > power and power != -1:
                        continue  # skip the neighbor if the power constraint is not met

                    queue.append(neighbor)  # Enqueues the neighbor

        return visited

    def get_path_with_power(self, source, destination, power=-1):
        """
        Provides, if possible, the shortest path for a given power
        source : node of start
        destination : node of arrival
        power : power constraint (by default, none)
        """

        list_visited = [] #
        list_of_paths = [] #list of all possible paths
        list_dist_paths = [] #corresponds to the path
        queue = []

        for component in self.connected_components():
            if source in component and destination in component:
                queue.append([source]) #initialization of the queue
                list_visited.append(source) 

                while queue != []:
                    print("la queue est", queue)
                    current_path = queue.pop()
                    last_node = current_path[-1]

                    if last_node == destination:
                        list_of_paths.append(current_path)
                        #list_dist_paths.append(len(current_path) - 1)
                    
                    else:
                        for neighbor in self.graph[last_node]:
                            print(neighbor, power, neighbor[1] <= power)
                            if neighbor[1] <= power:  # skip the neighbor if the power constraint is not met
                                print(neighbor, power)

                                if neighbor[0] not in list_visited:
                                    queue.append(current_path+[neighbor[0]])
                                    list_visited.append(neighbor[0])
                
                if len(list_of_paths) >= 1:
                    print(f"Chemins possibles {list_of_paths}")
                    return list_of_paths[0]
                
                #else : 
                    #index_shortest_path = list_dist_paths.index(min(list_dist_paths))
                    #return list_of_paths[0]
    
        return None

    
            

                


        """
        queue = [(source, [source], 0)] # queue is a triplet (node, path, min_power)
        list_of_paths = [] # list of all possible paths with power

        while queue:
            node, path, min_power = queue.pop(0)
            if node == destination and min_power >= power: # if we have reached the destination with the minimum power, add the path
                list_of_paths.append(path)
            else:
               for neighbor, edge_power, _ in self.graph[node]:
                    if neighbor not in path and min_power+edge_power >= power:
                        queue.append((neighbor, path+[neighbor], min(min_power, edge_power)))
                    
        if len(list_of_paths) == 0:
            return None
        elif len(list_of_paths) >= 1:
            #min_dist : yet to code
            return list_of_paths[min_dist]
        """



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    

    def minimum_distance(self,origin, destination, possible_paths=[]):
        


        

def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """


    with open(filename, "r") as file:
            n, m = map(int, file.readline().split())
            g = Graph(range(1, n+1))
            for _ in range(m):
                edge = list(map(int, file.readline().split()))
                if len(edge) == 3:
                    node1, node2, power_min = edge
                    g.add_edge(node1, node2, power_min) # will add dist=1 by default
                elif len(edge) == 4:
                    node1, node2, power_min, dist = edge
                    g.add_edge(node1, node2, power_min, dist)
                else:
                    raise Exception("Format incorrect")
    return g
    



