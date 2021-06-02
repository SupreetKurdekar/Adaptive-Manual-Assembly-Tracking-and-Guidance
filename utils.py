import cv2
import numpy as np 
import cv2.aruco as aruco
import glob
import png
import sys
from copy import deepcopy
import graph_utils as gUtils
import re
import copy

import networkx as nx

import time

import plotly.graph_objects as go


def button_state_update(button,ids,n_frames):
    # if button aruco id is in ids
    # update button count
    # else make button count = 0

    if button["aruco_id"] not in ids:
        button["frame_count"] += 1
    else:
        button["frame_count"] = 0
        button["pressed"] = False

    if button["frame_count"] > n_frames:
        button["frame_count"] = n_frames
        button["pressed"] = True

def region_empty(reg_dict,reg_id,image):

    pt1 = tuple(reg_dict[reg_id]["box"][0])
    pt2 = tuple(reg_dict[reg_id]["box"][1])
    # print(pt1)
    # print(pt2)
    # cv2.rectangle(cad,pt1,pt2,(0,255,0))

    pt3 = tuple(reg_dict[reg_id]["inner_top_corner"])
    pt4 = tuple(reg_dict[reg_id]["inner_top_corner"] + np.array([reg_dict[reg_id]["inner_rect_height"],reg_dict[reg_id]["inner_rect_width"]]))       
    # cv2.rectangle(cad,pt3,pt4,(255,0,0))
    # cv2.circle(cad,pt3,10,(0,0,0))
    # print(pt3)

    # we now have roi
    roi = image[pt3[1]:pt4[1],pt3[0]:pt4[0]].copy()
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray,50,150)
    contours, h = cv2.findContours(edges, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return len(contours) <= 0

def reg_state_update(reg_dict,image):
    reg_dict[0]['state_list'] = []
    reg_dict[0]['id_list'] = []
    for reg_id in range(1,8):
        if reg_dict[reg_id]['state'] != 'E':
            # do opencv stuff for checking if reg is empty
            if region_empty(reg_dict,reg_id,image):
                reg_dict[reg_id]["empty"] = True
                reg_dict[0]['state_list'].append(reg_dict[reg_id]['state'])
                reg_dict[0]['id_list'].append(reg_id)

            elif not region_empty(reg_dict,reg_id,image):
                reg_dict[reg_id]["empty"] = False

    reg_dict[0]["state_list"].sort()
    workspace_state = ','.join(reg_dict[0]["state_list"])
    reg_dict[0]["state"] = workspace_state

def system_update(reg_dict,button1,button2,image,parameters,aruco_dict,n_frames):
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    parameters = aruco.DetectorParameters_create()
    reg_state_update(reg_dict,image)
    gray = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    ## for checking
    for reg_id in reg_dict:
        cv2.putText(image,str(reg_dict[reg_id]['state']),tuple(reg_dict[reg_id]["center"]),font1,1,(0,255,0))

    # cv2.imshow("gray",gray)
    # cv2.waitKey(0)
    corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    button_state_update(button1,ids,n_frames)
    button_state_update(button2,ids,n_frames)

    return len(reg_dict[0]['state_list'])

def system_update_with_prev(reg_dict,button1,button2,image,parameters,aruco_dict,n_frames):

    prev_dict = copy.deepcopy(reg_dict)

    font1 = cv2.FONT_HERSHEY_SIMPLEX
    parameters = aruco.DetectorParameters_create()
    reg_state_update(reg_dict,image)
    gray = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    ## for checking
    for reg_id in reg_dict:
        cv2.putText(image,str(reg_dict[reg_id]['state']),tuple(reg_dict[reg_id]["center"]),font1,1,(0,255,0))

    # cv2.imshow("gray",gray)
    # cv2.waitKey(0)
    corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    button_state_update(button1,ids,n_frames)
    button_state_update(button2,ids,n_frames)

    return len(reg_dict[0]['state_list']),prev_dict

def system_update_with_prev_assembled(reg_dict,button1,button2,image,parameters,aruco_dict,n_frames,assembled_history):

    
    prev_dict = copy.deepcopy(reg_dict)

    font1 = cv2.FONT_HERSHEY_SIMPLEX
    parameters = aruco.DetectorParameters_create()
    reg_state_update(reg_dict,image)
    gray = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    ## for checking
    for reg_id in reg_dict:
        cv2.putText(image,str(reg_dict[reg_id]['state']),tuple(reg_dict[reg_id]["center"]),font1,1,(0,255,0))

    # cv2.imshow("gray",gray)
    # cv2.waitKey(0)
    corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    button_state_update(button1,ids,n_frames)
    button_state_update(button2,ids,n_frames)

    
    return len(reg_dict[0]['state_list']),prev_dict

def update_graph_infeasibility(state,G):

    G.nodes[state]["feasibility"] = False
    for node in G.neighbors(state):
        # G.nodes[node]["feasibility"] = False
        G.edges[state,node]["feasibility"] = False

        if G.nodes[node]["level"] > G.nodes[state]["level"]:
            com_parts = list(G.nodes[node]["parts"].difference(G.nodes[state]["parts"]))
            com_parts.sort()
            com_part_node = ''.join(com_parts)
            if G.has_edge(node,com_part_node):
                G.edges[node,com_part_node]["feasibility"] = False
        else:
            com_parts = list(G.nodes[state]["parts"].difference(G.nodes[node]["parts"]))
            com_parts.sort()

            com_part_node = ''.join(com_parts)
            if G.has_edge(node,com_part_node):
                G.edges[node,com_part_node]["feasibility"] = False
    # also update entire graph
    # any node with all but one edge infeasible
    # must also become infeasible


def hanging_node_locator(G):
    hanging_nodes = []
    wrong_nodes = []
    for node in G.nodes():
        my_level = G.nodes[node]["level"]

        lower = [lower_node for lower_node in G.neighbors(node) if G.nodes[lower_node]["level"] < G.nodes[node]["level"] and G.edges[lower_node,node]["feasibility"] == True and G.nodes[node]["feasibility"]==True]

        if G.nodes[node]["level"] > 1 and len(lower) == 0:
            hanging_nodes.append(node)
        elif G.nodes[node]["level"] > 1 and len(lower) == 1:
            wrong_nodes.append(node)

    return hanging_nodes,wrong_nodes

def hanging_node_feasibility_update(G,hang_nodes):
    for node in hang_nodes:
        update_graph_infeasibility(node,G)

def highlight_possibilities(color,region_dict,G):

    if len(region_dict[0]["state_list"]) == 1:
        region_dict[0]["state_list"].sort()
        wkspace_state = region_dict[0]["state_list"][0]
        try:
            level = G.nodes[wkspace_state]["level"]
        except:
            pass
        upper = []
        for node in G.neighbors(wkspace_state):
            
# and G.nodes[node]["level"]-G.nodes[wkspace_state]["level"]==1 and G.edges[node,wkspace_state]==True
            if G.nodes[node]["feasibility"]==True and G.nodes[node]["level"]==level+1:
                upper.append(node)

        feasible_pieces = []
        for node in upper:

            com_parts = list(G.nodes[node]["parts"].difference(G.nodes[wkspace_state]["parts"]))
            com_parts.sort()
            com_part_node = ''.join(com_parts)
            if gUtils.is_combination_feasible(wkspace_state,com_part_node,G):
                feasible_pieces.append(com_part_node)

        possible_regions = []
        for node in feasible_pieces:
            for key in region_dict:
                if region_dict[key]["state"] == node:
                    possible_regions.append(key)

        for reg_id in possible_regions:
            cv2.rectangle(color,tuple(region_dict[reg_id]["box"][0]),tuple(region_dict[reg_id]["box"][1]),(0,255,255),3)




























# this function never works
# update the next state feasibility checking function to
# account for hanging nodes
def hanging_node_cleaning(G):
    hanging_nodes,wrong_nodes = hanging_node_locator(G)
    if len(wrong_nodes) > 0:
        print("non anded edges found")
        print(wrong_nodes)
    while len(hanging_nodes) > 0:
        for node in hanging_nodes:
            update_graph_infeasibility(node,G)
        hanging_nodes,wrong_nodes = hanging_node_locator(G)

        if len(wrong_nodes) > 0:
            print("non anded edges found")
            print(wrong_nodes)





