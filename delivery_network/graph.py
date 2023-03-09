class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        
    


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




    def get_path_with_power(self, src, dest, power):

        queue = [(src, [src], 0)] # BFS queue: a tuple (node, path, min_power)
        list_visited = set() # nodes already visited by BFS
        
        while queue:
            node, path, min_power = queue.pop(0)

            if node == dest and min_power >= power: # if we have reached the destination with the minimum power, return the path
                return path
            visited.add(node)
            for neighbor, edge_power, _ in self.graph[node]:
                if neighbor not in list_visited and min_power+edge_power >= power:
                    queue.append((neighbor, path+[neighbor], min(min_power, edge_power)))
                    list_visited.add(neighbor)
        
        return [] # if no path with the minimum power is found



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    

    def min_power(self,origin,destination):
            start = 0
            length = len(self.list_of_powers)
            end = length-1
            if destination not in self.depth_search(origin):
                return None
            while start != end:
                mid = (start+end)//2
                if destination not in self.depth_search(origin, power=self.list_of_powers[mid]):
                    start = mid
                else:
                    end = mid
                if end-start == 1:
                    start=end
            return self.list_of_powers[end]


        

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
    



