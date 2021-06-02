# Import packages for data cleaning
import numpy as np
import re # For finding specific strings in the text
# Import packages for data visualization
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx

import itertools
import graph_utils as gUtils
import copy
import utils

from networkx.drawing.nx_agraph import graphviz_layout


# Custom function to create an edge between node x and node y, with a given text and width
def make_edge(x, y, color, width):
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = color),
                       mode      = 'lines')

def make_edge_trace_list(G,width,pos):
    # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in G.edges():

        char_1 = edge[0]
        char_2 = edge[1]
        x0, y0 = pos[char_1]
        x1, y1 = pos[char_2]
        
        if G.edges[edge]["feasibility"] == True:
            trace  = make_edge([x0, x1, None], [y0, y1, None],color = 'cornflowerblue',width = width)
        else:
            trace  = make_edge([x0, x1, None], [y0, y1, None],color = 'crimson',width = width)

        edge_trace.append(trace)

    return edge_trace

# Make a node trace
def make_node_trace(G,pos,marker_size = 10,text_size = 25):
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "middle center",
                            textfont_size = text_size,
                            mode      = 'markers+text',
                            hoverinfo = 'none',
                            marker    = dict(color = [],
                                            size  = [],
                                            line  = None))


    # For each node in midsummer, get the position and size and add to the node_trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        if(G.nodes[node]["feasibility"] == True):
            node_trace['marker']['color'] += tuple(['cornflowerblue'])
        else:
            node_trace['marker']['color'] += tuple(['crimson'])
        node_trace['marker']['size'] += tuple([marker_size])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])

    return node_trace

def new_pos_from_old_pos(G,pos):

    height = 1000

    for node in G.nodes():
        if G.nodes[node]["level"] == 7:
            top_node = node
            break

    top = pos[top_node]

    width = 400
    # delta = 0.1 * width
    new_pos = {}
    new_pos[top_node] = top

    diffy = height/6

    for i,level_num in enumerate(reversed(range(7))):
        level_nodes = [node for node in G.nodes if G.nodes[node]["level"] == level_num]
        
        num_nodes = len(level_nodes)
        # width = width + delta
        
        difference = width/(num_nodes-1)
        x0 = top[0]
        init_x = x0-(width/2)
        
        y_coord = top[1]-i*diffy

        for id,node in enumerate(level_nodes):
            coord = list(pos[node])
            coord[0] = init_x + id*difference
            # coord[1] = top[1] - y_coord
            new_pos[node] = tuple(coord)
            

    return new_pos


def get_fig(G,marker_size,text_size,width):

    pos =  graphviz_layout(G, prog='dot')
    # print(pos)

    # scale = 10

    # center = pos['BLMRSSbSf']
    # pos = {}
    # for key in pos:
    #     pos[key] = np.asarray(pos[key]) - np.asarray(center)
    #     pos[key][0] = scale*pos[key][0]
    #     pos[key] = tuple(pos[key])
    # print(pos)

    pos = new_pos_from_old_pos(G,pos)
    

    node_trace = make_node_trace(G,pos,marker_size=marker_size,text_size=text_size)

    edge_trace = make_edge_trace_list(G,width=width,pos=pos)

    # print(node_trace)

    # Customize layout
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)', # transparent background
        plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
        yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
    )

    # Create figure
    fig = go.Figure(layout = layout)
    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    # Add node trace
    fig.add_trace(node_trace)
    # Remove legend
    fig.update_layout(showlegend = False)
    # Remove tick labels
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)

    return fig