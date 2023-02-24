class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    


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

        
        

        
        
        raise NotImplementedError
    

    def connected_components(self):
        list_component = []
        node_visited = {node:False for node in self.nodes}

        def dfs(node):

            component = [node]

            for neighbour in self.graph[node]:
                neighbour = neighbour[0]

                if not node_visited[neighbour]:
                    node_visited[neighbour] = True
                    component += dfs(neighbour)
            return component
        
        for node in self.nodes:
            if not node_visited[node]:
                list_component.append(dfs(node))
    
        return list_component



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        
        
        """
        Should return path, min_power. 
        """
        raise NotImplementedError


def graph_from_file(filename):
    
    with open(filename) as file:
        line1 = file.readline().split()
        n = int(line1[0])
        m = int(line1[1])
        nodes = [i for i in range(1, n+1)]
        G = Graph(nodes)
        for i in range(m):
            linei = file.readline().split()
            node1 = int(linei[0])
            node2 = int(linei[1])
            power_min = int(linei[2])

            if len(linei)>3:
                dist = int(linei[3])
                G.add_edge(node1, node2, power_min, dsit)

            else:
                G.add_edge(node1, node2, power_min)
            
    return G 


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
 