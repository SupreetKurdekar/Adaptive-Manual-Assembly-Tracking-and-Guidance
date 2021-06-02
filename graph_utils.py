import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout
import re

def get_node_attribute_dictionary(graph,attribute):
    """This function takes a networkx graph as an input and returns a dictionary
    with node as key and requested attribute as value
    
    Args:
        graph (nx graph): input graph
        attribute ([string]): that data stored in node of the graph

    Returns:
        dictionary: keys are nodes of the graph and values are the required attributes
    """
    node_attribute_dict = {node_name: data[attribute] for node_name,data in graph.nodes(data=True)}

    return node_attribute_dict

def infeasible_reject(node,feasibility_graph):
    return nx.is_connected(feasibility_graph.subgraph(node[1]["parts"]))

def get_AO_graph(parts,feasible_edges):

    # feasible fit graph
    # for ABCDE chain link system
    ff_graph = nx.Graph()
    ff_graph.add_edges_from(feasible_edges)
    # nx.draw(ff_graph, with_labels=True, arrows=False)

    # parts = parts.sorted()
    # print(parts)

    hierarchy = {}
    for r in range(1,len(parts)+1):
        inner_list = list(itertools.combinations(parts, r))
        hierarchy[r] = inner_list

    max_level = len(hierarchy)

    G = nx.Graph()

    for level in reversed(list(hierarchy.keys())):
        for node in hierarchy[level]:
            # node = set(node)
            nodeName = ""
            for s in node:
                nodeName += s
            # print(type(nodeName))
            G.add_node(nodeName,parts = node,level = level)

    # make a new graph
    G2 = nx.Graph()

    # convert all nodes to nodes with sets. 
    # add an attribute called anded_pairs
    # anded_pairs is a list of pairs of nodes which together form that node
    for node in G.nodes(data=True):
        G2.add_node(node[0],parts = set(node[1]["parts"]),level = node[1]["level"],anded_pairs = [])


    for level_num in range(1,max_level):
        # iterating through one level nodes in G2
        for node_name,node_data in G2.nodes(data=True):
            if node_data['level']==level_num:
                #iterating through all nodes above current level
                for upper_node_name,upper_node_data in G2.nodes(data=True):
                    if upper_node_data['level'] > level_num:
                        # making edges if lower is subset of upper node
                        if node_data["parts"].issubset(upper_node_data["parts"]):
                            G2.add_edge(node_name,upper_node_name)
                            # Each edge now has an trribute called anded which is currently set to False
                            G2[node_name][upper_node_name]['anded']=False


    # remove the geometrically infeasible nodes
    to_be_removed = []
    for node in G2.nodes(data=True):
        if not infeasible_reject(node,ff_graph):
            to_be_removed.append(node[0])
    G2.remove_nodes_from(to_be_removed)

    # # go through each edge of each node and remove each edge that does not have a complement
    # # if complement is found store complement edge as anding edges
    edges_to_be_removed = []

    for node_name,node_data in G2.nodes(data=True):
        for neighbour in G2.neighbors(node_name):
            if G2.nodes[neighbour]["level"] < node_data["level"]:
                anded_pair = []
                comp = node_data["parts"].difference(G2.nodes[neighbour]["parts"])
                comp_level = len(comp)
                for neighbour_2 in G2.neighbors(node_name):
                    if (neighbour_2 != neighbour) and (comp_level == G2.nodes[neighbour_2]["level"]) and (comp == G2.nodes[neighbour_2]["parts"]):
                        anded_pair.append(neighbour_2)
                        break
                if len(anded_pair)>0:
                    G2[node_name][neighbour]["anded"] = True
                    G2[node_name][anded_pair[0]]["anded"] = True
                    anded_pair.append(neighbour)
                    G2.nodes[node_name]['anded_pairs'].append(anded_pair)
                else:
                    edges_to_be_removed.append((node_name,neighbour))

    G2.remove_edges_from(edges_to_be_removed)

    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    nx.nx_agraph.write_dot(G2,'test.dot')

    # same layout using matplotlib with no labels
    pos=graphviz_layout(G2, prog='dot')
    nx.draw(G2, pos, with_labels=True, arrows=False,node_size=1000)

    plt.savefig("Graph.png", format="PNG")
    plt.show()


def does_combination_exist(name1,name2,G):

    list1 = re.findall('[A-Z][^A-Z]*',name1)
    list2 = re.findall('[A-Z][^A-Z]*',name2)

    sub_parts_list = list1 + list2
    sub_parts_list.sort()

    name = ''.join(sub_parts_list)
    
    return G.has_node(name)

def is_combination_feasible(name1,name2,G):

    list1 = re.findall('[A-Z][^A-Z]*',name1)
    list2 = re.findall('[A-Z][^A-Z]*',name2)

    sub_parts_list = list1 + list2
    sub_parts_list.sort()

    name = ''.join(sub_parts_list)
    
    ans1 = G.has_node(name)
    if ans1:
        c1 = G.nodes[name]["feasibility"]
    else:
        c1 = False
    ans2 = G.has_edge(name,name1)
    ans3 = G.has_edge(name,name2)


    if ans2 and ans3:
        c2 = G.edges[name,name1]["feasibility"] and G.edges[name,name2]["feasibility"]
    else:
        c2 = False
    
    return c2 and c1

# def get_subgraph_with_subassembly(my_parts,G):

#     #get list of nodes where my_parts is a subset of node.parts
#     for name,data in G.nodes(data=True):


def get_feasibility_marked_graph(parts,feasible_edges):

    # parts = ["A","B","C"]
    # feasible_edges = [("A","B"),("A","C")]
    # path_out = "/home/supreet/vision3/vision_realsense/graphStuff/graphs/three_part.gpickle"
    # parts = ["L","R","M","S","Sf","Sb","B"]
    # feasible_edges = [("L","B"),("R","B"),("L","R"),("S","Sf"),
    # ("Sf","Sb"),("M","L"),("M","R"),("M","L"),("Sb","R"),("Sb","L")]


    # parts = ["L","R","M","S","Sf","Sb"]
    # feasible_edges = [("L","R"),("S","Sf"),("Sf","L"),("Sf","R"),("Sf","Sb"),("M","L"),("M","R"),("Sb","L"),("Sb","L")]

    # feasible fit graph
    # for ABCDE chain link system
    ff_graph = nx.Graph()
    ff_graph.add_edges_from(feasible_edges)
    # nx.draw(ff_graph, with_labels=True, arrows=False)

    parts.sort()
    # print(parts)

    nx.nx_agraph.write_dot(ff_graph,'test.dot')

    # # same layout using matplotlib with no labels
    # pos=graphviz_layout(ff_graph, prog='dot')
    # nx.draw(ff_graph, pos, with_labels=True, arrows=False,node_size=2000,font_size=18)
    # plt.ylabel("Direction of assembly")
    # plt.show()

    hierarchy = {}
    for r in range(1,len(parts)+1):
        inner_list = list(itertools.combinations(parts, r))
        hierarchy[r] = inner_list

    max_level = len(hierarchy)

    G = nx.Graph()

    for level in reversed(list(hierarchy.keys())):
        for node in hierarchy[level]:
            # node = set(node)
            nodeName = ""
            for s in node:
                nodeName += s
            # print(type(nodeName))
            G.add_node(nodeName,parts = node,level = level)

    # make a new graph
    G2 = nx.Graph()

    # convert all nodes to nodes with sets. 
    # add an attribute called anded_pairs
    # anded_pairs is a list of pairs of nodes which together form that node
    for node in G.nodes(data=True):
        G2.add_node(node[0],parts = set(node[1]["parts"]),level = node[1]["level"],anded_pairs = [],feasibility = True)

    nx.nx_agraph.write_dot(G2,'test.dot')

    # # same layout using matplotlib with no labels
    # pos=graphviz_layout(G2, prog='dot')
    # nx.draw(G2, pos, with_labels=True, arrows=False,node_size=2000,font_size=18)
    # plt.ylabel("Direction of assembly")
    # plt.show()


    for level_num in range(1,max_level):
        # iterating through one level nodes in G2
        for node_name,node_data in G2.nodes(data=True):
            if node_data['level']==level_num:
                #iterating through all nodes above current level
                for upper_node_name,upper_node_data in G2.nodes(data=True):
                    if upper_node_data['level'] > level_num:
                        # making edges if lower is subset of upper node
                        if node_data["parts"].issubset(upper_node_data["parts"]):
                            G2.add_edge(node_name,upper_node_name,feasibility=True)
                            # Each edge now has an trribute called anded which is currently set to False
                            G2[node_name][upper_node_name]['anded']=False

    nx.nx_agraph.write_dot(G2,'test.dot')

    # # same layout using matplotlib with no labels
    # pos=graphviz_layout(G2, prog='dot')
    # nx.draw(G2, pos, with_labels=True, arrows=False,node_size=2000,font_size=18)
    # plt.ylabel("Direction of assembly")
    # plt.show()


    # remove the geometrically infeasible nodes
    to_be_removed = []
    for node in G2.nodes(data=True):
        if not infeasible_reject(node,ff_graph):
            to_be_removed.append(node[0])
    # G2.remove_nodes_from(to_be_removed)
    for node_name in to_be_removed:
        G2.nodes[node_name]["feasibility"] = False
        # make all edges connected to the node as false for feasibility
        for edge in list(G2.edges(node_name,data=True)):
            G2[edge[0]][edge[1]]["feasibility"] = False        

    # print(G2.edges(data=True))

    nx.nx_agraph.write_dot(G2,'test.dot')

    # # same layout using matplotlib with no labels
    # pos=graphviz_layout(G2, prog='dot')
    # nx.draw(G2, pos, with_labels=True, arrows=False,node_size=2000,font_size=18)
    # plt.ylabel("Direction of assembly")
    # plt.show()

    # # go through each edge of each node and remove each edge that does not have a complement
    # # if complement is found store complement edge as anding edges
    edges_to_be_removed = []

    # do not use infeasible nodes in the process of making anded pairs
    # do not make infeasible edges anded = True
    for node_name,node_data in G2.nodes(data=True):
        if G2.nodes[node_name]["feasibility"]:
            for neighbour in G2.neighbors(node_name):
                if G2.nodes[neighbour]["level"] < node_data["level"] and G2.nodes[neighbour]["feasibility"]:
                    anded_pair = []
                    comp = node_data["parts"].difference(G2.nodes[neighbour]["parts"])
                    comp_level = len(comp)
                    for neighbour_2 in G2.neighbors(node_name):
                        if (G2.nodes[neighbour_2]["feasibility"]) and (neighbour_2 != neighbour) and (comp_level == G2.nodes[neighbour_2]["level"]) and (comp == G2.nodes[neighbour_2]["parts"]):
                            anded_pair.append(neighbour_2)
                            
                            break
                    if len(anded_pair)>0:
                        G2[node_name][neighbour]["anded"] = True
                        G2[node_name][anded_pair[0]]["anded"] = True

                        if(G2[node_name][neighbour]["feasibility"] == False):
                            print("error")

                        if(G2[node_name][anded_pair[0]]["feasibility"] == False):
                            print("error")

                        anded_pair.append(neighbour)
                        anded_pair.sort()
                        G2.nodes[node_name]['anded_pairs'].append(anded_pair)
                    else:
                        edges_to_be_removed.append((node_name,neighbour))
                        G2[node_name][neighbour]["feasibility"] = False

    # G2.remove_edges_from(edges_to_be_removed)

    # # add sequential feasibility to edge
    # # remove 

    # currently graph nodes anded pairs contain double lists
    # removing them 
    for node in G2.nodes:
        G2.nodes[node]["anded_pairs"].sort()
        G2.nodes[node]["anded_pairs"] = list(G2.nodes[node]["anded_pairs"] for G2.nodes[node]["anded_pairs"],_ in itertools.groupby(G2.nodes[node]["anded_pairs"]))

    return G2

    



