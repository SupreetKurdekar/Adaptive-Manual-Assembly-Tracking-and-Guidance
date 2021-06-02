import cv2
import numpy as np 
import cv2.aruco as aruco
import glob
import png
import sys
from copy import deepcopy
import graph_utils as gUtils
import re
import utils
import networkx as nx
import pyrealsense2 as rs
import plotly.graph_objects as go

import networkx as nx
import pickle

import os
import config as user_config


assembled_history = {}
list_of_graphs = []

upper_white = 150
font = cv2.FONT_HERSHEY_SIMPLEX
n_frames = 15

button1 = {"name":"RightButton","pressed":False,"aruco_id":8,"frame_count":0}
button2 = {"name":"LeftButton","pressed":False,"aruco_id":11,"frame_count":0}


img_array = []

graph_list_storage_path = "/home/supreet/vision3/vision_realsense/graphStuff/graphs/expt1"
expt_name = "test_6.pkl"

vid_path = "/home/supreet/vision3/vision_realsense/scripts/state_recognition/expt6.avi"

# graph_path = "/home/supreet/vision3/vision_realsense/graphStuff/graphs/infeasible_graph.gpickle"

# ## read the graph into the script as G
# G = nx.readwrite.read_gpickle(graph_path)

parts = ["L","R","M","S","D","T","B"]
feasible_edges = [("L","B"),("R","B"),("L","R"),("S","D"),
("D","T"),("M","L"),("M","R"),("M","L"),("T","R"),("T","L")]

G = gUtils.get_feasibility_marked_graph(parts,feasible_edges)
# region dict state refers to the last state that it was in when it was filled
# this last state might be activated or deactivated based on currently fille/not filled 
# conditions
# region dict state refers to ownership

# it can also be updated when new sub assembly is added to it and empty state is switched off

region_dict = {0:{"name":"workspace","empty":True,"state":["E"],"state_list":[],"id_list":[],"area":0},
1:{"name":"1","empty":False,"state":"B","area":0},
2:{"name":"2","empty":False,"state":"D","area":0},
3:{"name":"3","empty":False,"state":"L","area":0},
4:{"name":"4","empty":False,"state":"R","area":0},
5:{"name":"5","empty":False,"state":"S","area":0},
6:{"name":"6","empty":False,"state":"M","area":0},
7:{"name":"7","empty":False,"state":"T","area":0}}



font = cv2.FONT_HERSHEY_SIMPLEX

num_boxes_to_be_found = 8
k = 0.15

# now keep rolling till you find 8 unique boxes and store their positions
# in region dict

aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
ar_dict2 = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

s = pipeline_profile.get_device().query_sensors()[1]
exposure = s.get_option(rs.option.exposure)
s.set_option(rs.option.exposure,500)

print("exposure",exposure)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1280, 960, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

while True:    

    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    color = np.asanyarray(color.get_data())

    color = cv2.rotate(color,cv2.ROTATE_180)

    img_size = np.array([color.shape[0],color.shape[1]])
    # cv2.imshow("flipped",color)
    # cv2.waitKey(0)
    # depth_file = path + 'depth/%s.png' % (Filename)
    # reader = png.Reader(depth_file)
    # pngdata = reader.read()
    # depth = np.array(tuple(map(np.uint16, pngdata[2])))
    cad = color.copy()
    # cad[depth == 0] = np.array([0,0,0],dtype = np.uint8)
    gray = cv2.cvtColor(cad, cv2.COLOR_BGR2GRAY)


    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    
    my_corners = np.array(corners).squeeze()
    my_centers = np.mean(my_corners,axis=1)

    my_ids = np.array(ids).squeeze()

    # getting repeated ids and their indices for extracting centers to make boxes
    unq, unq_idx, unq_cnt = np.unique(my_ids, return_inverse=True, return_counts=True)
    cnt_mask = unq_cnt > 1

    ## ids of regions that have been found
    dup_ids = unq[cnt_mask]

    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])

    # locations of same ids in my_centers
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])

    region_mask = np.zeros(cad.shape)
    # get pairs of points
    # boxes is a list, where each element is alist of two arrays 
    boxes = {}
    for region_id,locations in zip(dup_ids,dup_idx):
        boxes[region_id] = np.array([my_centers[i] for i in locations])

    
    num_boxes_found = len(boxes)
    print(num_boxes_found)

    if num_boxes_found == num_boxes_to_be_found:
        for region_id in boxes:
            region_dict[region_id]["box"] = boxes[region_id]
        
        break

# region marking has been done

# for each region get interior rectangle

for region_id in region_dict:

    box = region_dict[region_id]["box"]
    area = np.linalg.norm(box[0]-box[1])
    top_corner = np.min(box,axis=0).astype(int)

    h,w = (np.abs(box[0]-box[1])).astype(int)
    h1,w1 = (k*np.abs(box[0]-box[1])).astype(int)

    hk = h-2*h1
    wk = w - 2*w1

    new_top = top_corner + (k*np.abs(box[0]-box[1])).astype(int)
    region_dict[region_id]["area"] = area
    region_dict[region_id]["inner_top_corner"] = new_top
    region_dict[region_id]["inner_rect_width"] = wk
    region_dict[region_id]["inner_rect_height"] = hk
    region_dict[region_id]["center"] = np.mean(region_dict[region_id]["box"],axis=0)

prev_dict = deepcopy(region_dict)

list_of_graphs.append(deepcopy(G))

nodes_to_remove = []
for node in G.nodes:
    if G.nodes[node]["feasibility"] == False:
        nodes_to_remove.append(node)
G.remove_nodes_from(nodes_to_remove)

edges_to_remove = []
for edge in G.edges:
    if G.edges[edge]["feasibility"] == False:
        edges_to_remove.append(edge)
G.remove_edges_from(edges_to_remove)


list_of_graphs.append(deepcopy(G))

# main loop

# for Filename in range(len(glob.glob1(path,"*.jpg"))):
#     print(Filename)
#     img_file = path + '/%s.jpg' % (Filename)

img_center = img_size/2
img_center = img_center.astype(int)
img_center1 = deepcopy(img_center)
img_center[1] = img_center[1] + 45

left_button_pressed = False
while True:

    if(k==ord('q')):
        break
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    color = np.asanyarray(color.get_data())

    color = cv2.rotate(color,cv2.ROTATE_180)

    img_array.append(color)

    l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)
    # print(l)

    while(l == 0):

        if(k==ord('q')):
            break

        message = "Please select an object"
        cv2.putText(color,message,tuple(img_center),font,1,(255,0,0),1)
        cv2.imshow("display",color)
        k = cv2.waitKey(1)

        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        color = cv2.rotate(color,cv2.ROTATE_180)
        img_array.append(color)
        l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)


    while(l == 1):
        if(k==ord('q')):
            break
        # print("l = 1")

        message = "Please select another object"
        cv2.putText(color,message,tuple(img_center),font,1,(255,0,0),1)

        cv2.imshow("display",color)
        k = cv2.waitKey(1)

        if(left_button_pressed):

            # updating the graph with infeasibility information
            print("region_dict")
            state = region_dict[0]["state"]
            utils.update_graph_infeasibility(state,G)
            hanging_nodes,wrong_nodes = utils.hanging_node_locator(G)

            utils.hanging_node_feasibility_update(G,hanging_nodes)

            list_of_graphs.append(deepcopy(G))

            # print("feasible nodes in G")
            # fnodes = [x for x,y in G.nodes(data=True) if y['feasibility']==True]
            # print(len(fnodes))
            # print("feasible edges in G")
            # fedges = [x for x,y in G.edges(data=True) if y['feasibility']==True]
            # print(len(fedges))

            # recognising the bad region
            # the single assembly left is the bad assembly
            # now query assembled history to find previous locations of pieces
            # that form the current assembly

            bad_state_prev_dict = assembled_history[region_dict[0]["state"]]

            wksp_state_list = bad_state_prev_dict[0]["state_list"]
            disassembly_storage_regions = []
            for state in wksp_state_list:
                for key in bad_state_prev_dict:
                    if bad_state_prev_dict[key]["state"] == state:
                        disassembly_storage_regions.append([key,state])


            reg_1 = disassembly_storage_regions[0][0]
            reg_2 = disassembly_storage_regions[1][0]

            region_dict[reg_1]["state"] = deepcopy(bad_state_prev_dict[reg_1]["state"])
            
            # region_dict[reg1]["state"] = 
            region_dict[reg_2]["state"] = deepcopy(bad_state_prev_dict[reg_2]["state"])

            left_button_pressed = False

            try:
                while(region_dict[reg_1]["empty"] == True or region_dict[reg_2]["empty"] == True):

                    if(k==ord('q')):
                        break
                    message = "Dissassemble parts"
                    cv2.putText(color,message,tuple(img_center),font,1,(255,0,0),1)
                    cv2.putText(color,str(bad_state_prev_dict[reg_1]["state"]),tuple(bad_state_prev_dict[reg_1]["inner_top_corner"]),font,2,(255,0,0),1)
                    cv2.putText(color,str(bad_state_prev_dict[reg_2]["state"]),tuple(bad_state_prev_dict[reg_2]["inner_top_corner"]),font,2,(255,0,0),1)
                    cv2.rectangle(color,tuple(bad_state_prev_dict[reg_1]["box"][0]),tuple(bad_state_prev_dict[reg_1]["box"][1]),(0,0,255),1)
                    cv2.rectangle(color,tuple(bad_state_prev_dict[reg_2]["box"][0]),tuple(bad_state_prev_dict[reg_2]["box"][1]),(0,0,255),1)

                    ## marking regions with blue
                    ## and with their names


                    
                    cv2.imshow("display",color)
                    k = cv2.waitKey(1)

                    frames = pipeline.wait_for_frames()
                    color = np.asanyarray(frames.get_color_frame().get_data())
                    color = cv2.rotate(color,cv2.ROTATE_180)
                    img_array.append(color)
                    l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)

                print("stuff 2")


                frames = pipeline.wait_for_frames()
                color = np.asanyarray(frames.get_color_frame().get_data())
                color = cv2.rotate(color,cv2.ROTATE_180)
                img_array.append(color)
                l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)
    
                cv2.imshow("display",color)
                k = cv2.waitKey(1)

                print("stuff")       
            except:
                pass



        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        color = cv2.rotate(color,cv2.ROTATE_180)
        img_array.append(color)
        l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)

        # highlighting the regions that can be selected based on the state of the workspace and regions

        utils.highlight_possibilities(color,region_dict,G)

    while(l == 2):

        if(k==ord('q')):
            break

        check2 = gUtils.is_combination_feasible(region_dict[0]["state_list"][0],region_dict[0]["state_list"][1],G)

        if check2:
            message = "This combination seems feasible"
            cv2.putText(color,message,tuple(img_center),font,1,(255,0,0),1)

            if button1["pressed"]:
                message = "Connected!"
                cv2.putText(color,message,tuple(img_center),font,1,(255,0,0),1)

                ###
                region_dict[0]['state_list'].sort()
                # temp_state = ''.join(region_dict[0]['statelist'])
                
                part_list1 = re.findall('[A-Z][^A-Z]*', region_dict[0]["state_list"][0])
                part_list2 = re.findall('[A-Z][^A-Z]*', region_dict[0]["state_list"][1])
                part_list1.extend(part_list2)
                part_list1.sort() 
                combined_state = ''.join(part_list1)
                assembled_history[combined_state] = deepcopy(region_dict)

                region_dict[0]["state"] = combined_state
                ## now wkspace has only one object so update that in the state_list
                region_dict[0]['state_list'] = [region_dict[0]["state"]]

                id1 = region_dict[0]['id_list'][0]
                id2 = region_dict[0]['id_list'][1]

                if region_dict[id1]['area'] > region_dict[id2]['area']:

                    region_dict[id1]["state"] = region_dict[0]['state']

                    region_dict[id2]["state"] = 'E'

                else:
                    region_dict[id2]["state"] = region_dict[0]['state']

                    region_dict[id1]["state"] = 'E'

            elif button2["pressed"]:

                while(l==2):

                    if(k==ord('q')):
                        break
                    left_button_pressed = True
                    message = "Infeasible sequence found" 
                    message2 = "Please return the correct part" 
                    cv2.putText(color,message,tuple(img_center1),font,1,(0,0,255),2)
                    cv2.putText(color,message2,tuple(img_center),font,1,(255,0,0),1)

                    # highlight the two regions fromwhere the current two parts came
                    # show them as 

                    cv2.imshow("display",color)
                    k = cv2.waitKey(1)

                    frames = pipeline.wait_for_frames()
                    color = np.asanyarray(frames.get_color_frame().get_data())
                    color = cv2.rotate(color,cv2.ROTATE_180)
                    img_array.append(color)
                    l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)
                    # print(l)


        else:
            message = "This combination is infeasible"
            cv2.putText(color,message,tuple(img_center),font,1,(0,0,255),1)

            



        cv2.imshow("display",color)
        k = cv2.waitKey(1)

        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        color = cv2.rotate(color,cv2.ROTATE_180)
        img_array.append(color)
        l,prev_dict = utils.system_update_with_prev(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)


    while(l > 2):
        
        if(k==ord('q')):
            break
        # print("l > 2")
        message = "Please Keep only two objects in workspace"
        cv2.putText(color,message,tuple(img_center),font,1,(0,0,255),1)
        cv2.imshow("display",color)
        k = cv2.waitKey(1)



        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        color = cv2.rotate(color,cv2.ROTATE_180)
        img_array.append(color)
        l = utils.system_update(region_dict,button1,button2,color,parameters,ar_dict2,n_frames)
        # print(l)

height, width, layers = color.shape
size = (width,height)

out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

import pickle

with open(os.path.join(graph_list_storage_path,expt_name), 'wb') as f:
    pickle.dump((list_of_graphs), f)