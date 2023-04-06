import time
import random

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.list_edges = []
        self.dict_edges = {}
        self.parents = {}
        self.depths = {}
        


    def dfs(self, node, node_visited):
        """
        This function is a depth-first-search algorithm.
        Parameters : 
        node : the starting node of the DFS
        node_visited : a dict representing if a given node (key of the dictionary) has been visited or not (value of the dictionary is a Boolean)
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

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items(): 
                self.dict_edges[source] = destination  #we create a dictionary where each key is the node of source, and the value are the possible paths
                output += f"{source}-->{destination}\n"
 
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        
        
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
        self.nb_edges += 1
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))

        self.list_edges.append((node1, node2, power_min))
    
    def path_distance(self, path):
        """
        This function returns the distance of a given path
        """
        total_distance = 0
        for i in range(len(path)-1): 
            for neighbor in self.graph[path[i]]:
                if neighbor[0] == path[i+1]:
                    total_distance += neighbor[2] #update the distance  
                    break
        return total_distance

    def power_required_for_path(self, path):
        """
        This function returns the power required to achieve a path
        """
        power_required_for_the_path = 0
        for i in range(len(path)-1):
            for neighbor in self.graph[path[i]]:
                if neighbor[0] == path[i+1]:
                    power_required_for_the_path = max(power_required_for_the_path, neighbor[1]) #power required for the path is the min of max
                    break
        return power_required_for_the_path

    
    def all_possible_paths(self, source, destination):
        """
        Arguments : 
        - source 
        - destination

        Returns : 
        list_of_paths : list of tuples. Each tuple is composed of :
            - one path between source and destination
            - distance of the given path
            - power required to achieve the given path 

        Please not that this works for small networks only, as it is not optimized at all
        Example  of output :
        all_possible_paths(1,2): 
        """
 
        list_of_paths = []
        list_of_distances = []
        path_dist_power = []

        for component in self.connected_components():
            if source in component and destination in component: #if there is a connection between source and destination
                queue = [[source]]
                list_visited = []

                while queue != []: #we explore all possible paths, meaning we explore all nodes
                    current_path = queue.pop() #
                    last_node = current_path[-1]

                    if last_node == destination: 
                        power_required_for_the_path = 0
                        for i in range(len(current_path)-1):
                            for neighbor in self.graph[current_path[i]]:
                                if neighbor[0] == current_path[i+1]:
                                    power_required_for_the_path = max(power_required_for_the_path, neighbor[1])
                        list_of_paths.append(current_path)
                        list_of_distances.append(self.path_distance(current_path))
                        path_dist_power.append((current_path, self.path_distance(current_path), power_required_for_the_path))
                    else:
                        for neighbor in self.graph[last_node]:
                            if neighbor[0] not in current_path: #we don't go over the same node twice
                                queue.append(current_path+[neighbor[0]]) 
                                list_visited.append(neighbor[0]) #this node has been visited

        print(list_of_distances)
        print(list_of_paths)
        print(path_dist_power)
        return path_dist_power

    def get_path_with_power(self, source, destination, power=-1):
        """
        Provides, if possible, the shortest path for a given power
        source : node of start
        destination : node of arrival
        power : power constraint (by default, none)
        """
        paths = self.all_possible_paths(source, destination)
        print(paths)
        valid_paths = []
        for path in paths:
            if power == -1 or path[2] <= power:
                valid_paths.append(path) #creating the list of valid paths

        if not valid_paths:
            return None #if there are only paths with power_required > power
        

        shortest_path_index = 0
        
        for i in range(0, len(valid_paths)):
            if valid_paths[i][1] < valid_paths[shortest_path_index][1]:
                shortest_path_index = i
        shortest_path = valid_paths[shortest_path_index][0]
        
        return shortest_path

    def min_power(self, source, destination):
        """
        Provides, if possible, the shortest path for a given power
        source : node of start
        destination : node of arrival
        power : power constraint (by default, none)
        """
        
        paths = self.all_possible_paths(source, destination)
        print(paths)
        shortest_path_index = 0
        
        for i in range(0, len(paths)):
    
            if paths[i][2] < paths[shortest_path_index][2]:
                shortest_path_index = i #given that we have generated every possible path and their distance, we just take the shortest one
        
        shortest_path = paths[shortest_path_index][0]
        min_power = paths[shortest_path_index][2]
        
        return (shortest_path, min_power)

    def estimate_time(self,filename):
        """
        This fonction is meant to estimate the performance of our code. 
        """
        
        with open(filename, "r") as file:
            n=self.nb_nodes           
            src,dest=random.sample(self.nodes,2)    #we take two random nodes   
            if self.min_power(src, dest)!=None:
                start_time= time.perf_counter()                
                self.min_power(src,dest)
                end_time= time.perf_counter()                
                time_path=(end_time - start_time)*n #we multiply by the number of nodes
            else:
                return None
        return time_path

    
    def dfs_parents(self, node, visited, parents, depths):
        depth=0
        parents
        
        for neighbour,puissance,distance in self.dict_edges[node]:
            visited.append(node)
            depth+=1
            if neighbour not in visited:
                parents[neighbour]=(node,puissance)
                depths[neighbour]=(depth)
                dfs_parents_and_depths(self, neighbour, visited, parents,depths)
        print("parents",parents)
        print("depths", depths)
        self.parents = parents
        self.depths = depths

    def min_power_kruskal(g_mst, start, dest):
        """
        This function returns the shortest path between two nodes in a graph, using the depths of the start and destination nodes in the mst tree 
        """


        if start in g_mst.connected_components and dest in g.mst_connected_components:
            parents, depths = dfs_parents(g_mst, 1, [], {1:(1,0)}, {1 : 0}) #we initialize the root of the mst as the first node
            path_from_start = []
            path_from_dest = []
            final_path = [] 

            while depths[start] > depths[dest]:
                path_from_start.append(start)
                start = parents[start]

            while depths[dest] > depths[dest]:
                path_from_dest.append(dest)
                dest = parents[depth]

            while start != end:
                path_from_start.append(start)
                path_from_dest.append(end)
                start = parents[start]
                end = parents[end]

            reverse_path = path_from_dest.reverse() #as we started from the end, we have to reverse the path_from_dest
            reverse_path.pop(0) #we remove the first node of path_from_dest, which is the same as the last element of the path_from_start
            
            final_path = path_from_start + reverse_path
            return final_path, power_min

        else : #Start and Destination are not connected 
            
            return None
        


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
    print(g)
    return g
 

class UnionFind():
    """
    As suggested in the paper, we use the Union Find data structure.
    It is a direct application of the Dasgupta et al book
    """

    def __init__(self):
        self.parent = None
        self.rank = None

    def makeset(self):
        self.parent = self
        self.rank = 0

    def find(self):
        if self != self.parent:
            self = self.parent
        return self
    
    def union(self, other):
        root_self = self.find()
        root_other = other.find()
        if root_self == root_other:
            return 
        if root_self.rank > root_other.rank:
            root_other.parent = root_self
        else:
            root_self.parent = root_other
            if root_self.rank == root_other.rank:
                root_other.rank = root_other.rank + 1


def kruskal(g):
    """
    This function returns a minimum spanning tree of a given graph
    We are referring to the Dasgupta et al book https://people.eecs.berkeley.edu/~vazirani/algorithms/chap5.pdf
    """

    list_edges = g.list_edges
    edges_sorted = sorted(list_edges, key=lambda x: x[2]) #we sort the edges by their power  
    g_mst = Graph(g.nodes)
    mst_dict = {}
    mst_set = []

    for node in g.nodes: 
        mst_dict[node] = UnionFind() 
        mst_dict[node].makeset()

    for edge in edges_sorted:
        
        node1, node2, min_power = edge[0], edge[1], edge[2]

        if mst_dict[node1].find() != mst_dict[node2].find() :
            mst_set.append((node1, node2, min_power))
            mst_dict[node1].union(mst_dict[node2])
    

    for edge in mst_set:
        source, destination, power = edge[0], edge[1], edge[2]
        g_mst.add_edge(source, destination, power)  #we add the 'elected' edges to the mst
    
    
    print("End", g_mst)
    print("mst dict of nodes is", g_mst.dict_edges)
    g_mst.dfs_parents(1, [], {1:(1,0)}, {1:0} )

    return g_mst


def routes_out_from_file(filename,filename2,powers=[]):
    """
    This function reads a given file and returns an output file containing the minimum power needed to go from a start to a destination
    """

    g=graph_from_file(filename2)
    k=kruskal(g)   

    with open(filename,"r") as file:
        n=int(file.readline())
        print(n)

        for _ in range(1,n+1):
            trajets=list(map(int,file.readline().split())) 
            k.parents_and_depths(trajets[0], [], {trajets[0]:(trajets[0],0)}, {trajets[0]:0},depth=0)
            a=k.min_power_kruskal(trajets[0],trajets[1],[],[],[],0)
            powers.append(a[1])

    f=open('routes.x.out','w',encoding='utf-8')

    for k in range(1,n+1):
        f.write(str(powers[k-1])+"\n")
    
    f.close()

    return f


class Catalogue:
    """
    This class is used to create a catalogue of trucks and paths. It stores :
    - the power and cost of each truck 
    - the min_power and profit of each path.
    """
   
    def __init__(self,trucks=[],paths=[]):
        self.trucks=trucks
        self.paths=paths
        self.catalogue_trucks = dict([(n, []) for n in trucks])
        self.nb_trucks = len(trucks)
        self.nb_paths=len(paths)
        self.catalogue_paths=dict((m,[])for m in paths)
        
    def __str__(self):
        """Prints two catalogues. One as a list of the power and the cost for each truck (one per line) and one as a list of the min_power and the profit for each path"""
        if not self.catalogue_trucks or not self.catalogue_paths:
            output = "The catalogue is empty"            
        else:
            output = f"The catalogue has {self.nb_trucks} trucks available and {self.nb_paths} paths possible"
        return output
    
    
    def add_truck(self, truck,power,cost):
        self.catalogue_trucks[truck]=(power,cost)
        
    def add_path(self, path,min_power,profit):
        self.catalogue_paths[path]=(min_power,profit)
        
    def trucks_for_path(self,path):
        trucks_for_path=[]
        for i in range(1,len(self.trucks)+1):
            truck=i
            
            if self.catalogue_paths[path][0]<=self.catalogue_trucks[truck][0]:
                trucks_for_path.append(truck)
                
            trucks_for_path = sorted(trucks_for_path, key=lambda truck: self.catalogue_trucks[truck][1])
        return trucks_for_path

     

def catalogue_from_file(filename1,filename2):
    """
    This function reads two files and returns a catalogue of trucks and paths
    """
    with open(filename1, "r") as f1, open(filename2, "r") as f2:
        n = int(f1.readline())
        m = int(f2.readline())
        
        c = Catalogue(tuple(range(1, n+1)),tuple(range(1,m+1)))
   
        for i in range(1,n+1):
            truck = list(map(int,f1.readline().split())) #we read the file line by line and we store the power and the cost of each truck in a list
            if len(truck) == 2:
                power=truck[0] #power of the truck
                cost=truck[1] #cost of the truck
                c.add_truck(i,power,cost)
                
            else:
                raise Exception('Format incorrect')
        
        for i in range(1,m+1):
            path = list(map(int,f2.readline().split())) 
            if len(path)==2:
                min_power=path[0]
                profit=path[1]
                c.add_path(i,min_power,profit)  #we add the path to the catalogue
            else :
                raise Exception("Format incorrect")
        return c 

def knapsack(c,solution_optimale={}):
    """
    This function returns the utility and the cost of the optimal solution.
    Input : a catalogue of trucks and paths
    Output : the utility and the cost of the optimal solution
    """

    paths_sorted = sorted(c.paths, key=lambda path: c.catalogue_paths[path][1],reverse=True)
    utility=0
    total_cost=0
    for path in paths_sorted:
        truck=c.trucks_for_path(path)[0]
        if total_cost + c.catalogue_trucks[truck][1]>25*(10**9): #if we don't have the budget
            break
        else:
            solution_optimale[path]=truck  #we add the path to the solution
            utility+=c.catalogue_paths[path][1] #we add the profit of the path to the utility
            total_cost+=c.catalogue_trucks[truck][1] #we add the cost of the truck to the total cost
    return utility, total_cost,solution_optimale 
