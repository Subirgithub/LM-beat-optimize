#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:58:47 2023

@author: subir.swapan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:06:35 2023

@author: subir.swapan
"""
# Import packages
import pandas as pd
import numpy as np
import os
from geopy import distance
import networkx as nx
import warnings

os.chdir('/Users/subir.swapan/Desktop/Large /LM beats/outputs')
#Read all input files
 
# input_file=pd.read_csv("lm_pincode_units_sep23_v2.csv")   #input_file.columns input_file.dtypes
input_file=pd.read_csv("lm_pincode_units_mar24.csv") 
# avg_cft_vertical=pd.read_csv("CFT_per_vertical.csv")
# input_file['units'] = input_file['units'].astype('float64')

pincode_geo_details =pd.read_csv("pincode_geo_details.csv")
mdm_dump =pd.read_csv("mdm_dump.csv")
mdm_dump=mdm_dump[["name","latitude","longitude"]]
mdm_dump['name']=mdm_dump['name'].str.lower()
# ist_by_pincode =pd.read_csv("dghr_hub_pincodes_ist.csv")
ist_by_pincode =pd.read_csv("ist_by_pincode_design.csv") #pd.read_csv("ist_by_pincode.csv")
# orderid_vertical_map =pd.read_csv("orderid_vertical_map_june.csv")
# orderid_vertical_map = orderid_vertical_map[['order_external_id', 'analytic_vertical']].drop_duplicates()
ds_pbh_map=pd.read_csv("lm_pbh_mar24.csv")
#provide inputs for model here
#HUb to run the code
hub_list=["SATELLITEHUB_LUD1","SATELLITEHUB_FKBDQ","SATELLITEHUB_BDQ","SATELLITEHUB_HOSF1","SATELLITEHUB_FKGTR","SATELLITEHUB_AMD1","SATELLITEHUB_BBK","SATELLITEHUB_PTA2","SATELLITEHUB_FKKRM","SATELLITEHUB_AGR1"] 
# hub_list=["SatelliteHub_SLG","Satellitehub_TSK","SATELLITEHUB_AZM2","SATELLITEHUB_JNP2","Satellitehub_JAJ","Satellitehub_GUW2","Satellitehub_BHD","Satellitehub_MRD",
#           "Satellitehub_FKBRM","Satellitehub_BET1","SatelliteHub_MCL","Satellitehub_ARY","SatelliteHub_BLS","SATELLITEHUB_NGAN",
#           "SatelliteHub_GJL","Satellitehub_NPT","Satellitehub_DND"]
# hub="SATELLITEHUB_BDQ1"
# hub_list =pd.read_csv("hub_list_2.csv")
# hub_list = hub_list["hub_name"].values.flatten().tolist()
# hub_list=["Satellitehub_FKTSK"]
hub_ipp_list=pd.read_csv("hub_ipp_list.csv")
# report_run_date = pd.to_datetime('2023-07-02').date()

van_avg_speed=25 #km/h
cft_factor=1  #max 90% utilization
Van_ftl_cft=180*cft_factor
Van_oft_hrs_max=12
van_dist_max=200
dist_factor=1.2
# start_date = pd.to_datetime('2023-07-02').date()
# end_date = pd.to_datetime('2023-07-08').date()
# file_name = "pentaho_report_"

def christofides_shortest_path(points, dist_matrix, start_point):
    # Step 1: Construct a complete graph from the distance matrix for the subset of points
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u = points[i]
            v = points[j]
            G.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 2: Construct a minimum spanning tree (MST) from the complete graph
    mst = nx.minimum_spanning_tree(G)

    # Step 3: Create a graph with odd degree vertices from the MST
    odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]
    odd_degree_subgraph = G.subgraph(odd_degree_nodes)

    # Step 4: Find a minimum weight perfect matching in the odd degree subgraph
    perfect_matching = nx.algorithms.matching.max_weight_matching(odd_degree_subgraph)

    # Step 5: Combine the MST and the perfect matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    for u, v in perfect_matching:
        multigraph.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 6: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=start_point))

    # Step 7: Traverse the Eulerian circuit and visit each node only once
    visited = set()
    path = []
    for u, v in eulerian_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)

    # Step 8: Calculate the total distance of the path
    total_distance = sum(dist_matrix[u][v] for u, v in zip(path, path[1:]))
    path.append(path[0])  # Complete the cycle

    return path, total_distance

def new_christofides_shortest_path(points, dist_matrix, start_point):
    # Step 1: Construct a complete graph from the distance matrix for the subset of points
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u = points[i]
            v = points[j]
            G.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 2: Construct a minimum spanning tree (MST) from the complete graph
    mst = nx.minimum_spanning_tree(G)

    # Step 3: Create a graph with odd degree vertices from the MST
    odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]
    odd_degree_subgraph = G.subgraph(odd_degree_nodes)

    # Step 4: Find a minimum weight perfect matching in the odd degree subgraph
    perfect_matching = nx.algorithms.matching.max_weight_matching(odd_degree_subgraph)

    # Step 5: Combine the MST and the perfect matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    for u, v in perfect_matching:
        multigraph.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 6: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=start_point))

    # Step 7: Traverse the Eulerian circuit and visit each node only once
    visited = set()
    path = []
    for u, v in eulerian_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)

    # Ensure the path ends at the start_point
    path.append(start_point)

    # Step 8: Calculate the total distance of the path
    total_distance = sum(dist_matrix[u][v] for u, v in zip(path, path[1:]))

    return path, total_distance

    # Step 1: Construct a complete graph from the distance matrix for the subset of points
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u = points[i]
            v = points[j]
            G.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 2: Construct a minimum spanning tree (MST) from the complete graph
    mst = nx.minimum_spanning_tree(G)

    # Step 3: Create a graph with odd degree vertices from the MST
    odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]
    odd_degree_subgraph = G.subgraph(odd_degree_nodes)

    # Step 4: Find a minimum weight perfect matching in the odd degree subgraph
    perfect_matching = nx.algorithms.matching.max_weight_matching(odd_degree_subgraph)

    # Step 5: Combine the MST and the perfect matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    for u, v in perfect_matching:
        multigraph.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 6: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=start_point))

    # Step 7: Traverse the Eulerian circuit and visit each node only once
    visited = set()
    path = []
    for u, v in eulerian_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)

    # Step 8: Calculate the total distance of the path
    total_distance = sum(dist_matrix[u][v] for u, v in zip(path, path[1:]))
    path.append(path[0])  # Complete the cycle

    return path, total_distance


def run_model(hub,input_file):
    
    pinlist_raw = input_file.loc[input_file['CurrentHubName'].str.lower() == hub.lower()].copy()
    pinlist_raw['CurrentHubName']=pinlist_raw['CurrentHubName'].str.lower()
    # pinlist_raw=input_file.copy()
    # pinlist_raw = pinlist_raw.rename(columns={'hub_name': 'CurrentHubName','pincode': 'CustomerPincode'})
    pinlist_raw = pd.merge(pinlist_raw, pincode_geo_details[['CustomerPincode', 'pincode_latitude', 'pincode_longitude']], on='CustomerPincode').drop_duplicates(subset='CustomerPincode')
    pinlist_raw = pd.merge(pinlist_raw, mdm_dump[['name', 'latitude', 'longitude']], left_on='CurrentHubName', right_on='name').drop(columns=['name'])
    # pinlist_raw = pd.merge(pinlist_raw, mdm_dump[['name', 'latitude', 'longitude']],how="left", left_on='CurrentHubName', right_on='name')
    pinlist_raw = pinlist_raw.rename(columns={'latitude': 'hub_latitude', 'longitude': 'hub_longitude'})
    
    pinlist_raw=pinlist_raw.sort_values(by=['units'], ignore_index=True, ascending=False)
        
    pinlist_raw = pinlist_raw[(pinlist_raw['pincode_latitude'] != -1) | (pinlist_raw['pincode_latitude'] != -1)].reset_index(drop=True)
    pinlist_raw = pinlist_raw[(pinlist_raw['hub_latitude'] != 0) | (pinlist_raw['hub_longitude'] != 0)].reset_index(drop=True)
    pinlist_raw = pinlist_raw.dropna(subset=['hub_latitude','hub_longitude'])

    
    pinlist_raw['distance_hub']=pinlist_raw.apply(lambda x: 
                                          distance.great_circle((x['hub_latitude'], x['hub_longitude']),(x['pincode_latitude'], x['pincode_longitude'])).km, axis=1)    
    
    pinlist_raw['distance_hub']=pinlist_raw['distance_hub']*dist_factor
    
    pinlist_raw = pinlist_raw[(pinlist_raw['cft_units'] > 0) & (pinlist_raw['distance_hub'] < 250)].reset_index(drop=True)    
    # pinlist_raw.to_csv('pin_dist.csv', index=False)
    #extended pincodes logic from analytics  
    # pinlist.loc[pinlist['distance_hub'] > 25, 'pincode_type'] = 'extended'
    # pinlist.loc[pinlist['distance_hub'] <= 25, 'pincode_type'] = 'local'
    pinlist=pinlist_raw.copy(deep = True) 
    # filtered_df = original_df[original_df['a'].isin([1, 2])]
    #Comment if you want to run for all local/extended pincodes
    # pinlist = pinlist[pinlist['pincode_type'] == 'local'].reset_index(drop=True)
    
    coord_arr = pinlist[['pincode_latitude', 'pincode_longitude']].to_numpy()    
    n = len(coord_arr)
    dist_matrix = np.zeros((n,n))    # initialize distance matrix to a square of zeros
    for i in range(n):
        for j in range(i, n):
            dist_matrix[i,j] = (distance.great_circle(coord_arr[i], coord_arr[j]).km)*dist_factor  # factor of 1.4 included to mimic road distance issues
            dist_matrix[j,i] = dist_matrix[i,j]       # for the symmetric part, no computation
    
    distance_df=pd.DataFrame(dist_matrix,  columns=pinlist.CustomerPincode.unique(), index=pinlist.CustomerPincode.unique())
    distance_df_var = distance_df.copy(deep = True)   
    output_list=[]
    output_list2=[]
    
    new_list=pinlist[['pincode_latitude', 'pincode_longitude']].copy(deep = True)  
    new_list.loc[len(new_list)] = [pinlist['hub_latitude'].values[0], pinlist['hub_longitude'].values[0]]
    coord_arr2=new_list.to_numpy()
    n = len(coord_arr2)
    dist_matrix2 = np.zeros((n,n))    # initialize distance matrix to a square of zeros
    for i in range(n):
        for j in range(i, n):
            dist_matrix2[i,j] = (distance.great_circle(coord_arr2[i], coord_arr2[j]).km)*dist_factor # factor of 1.5 included to mimic road distance issues
            dist_matrix2[j,i] = dist_matrix2[i,j]     
    distance_mat_short=pd.DataFrame(dist_matrix2)  
    
    del coord_arr,coord_arr2,dist_matrix,dist_matrix2,new_list,i,j,n
    #default beat numver
    pinlist["beat_number"]=99
    pinlist["units"]=pinlist["units"].astype('float64')
    #these need to be configured as per the Hub profile
    beat_number=1
    vehicle_count=0
    vehicles=pd.DataFrame(columns=['vehicle_count','pincodes','Shipments','Total_van_cft','van_oft_hrs','IST','van_travel_time','van_travel_dist'])  
    
    # hub_ipp=int((hub_ipp_list.loc[hub_ipp_list['hub'].str.lower() == hub.lower(),'IPP'].iloc[0])*1.1)     
    try:
     hub_ipp = int((hub_ipp_list.loc[hub_ipp_list['hub'].str.lower() == hub.lower(), 'IPP'].iloc[0]) * 1.1)
    except Exception:
     # print(f"Warning : Hub IPP not defined for {hub}: {e}")
     hub_ipp = 35
                      
    n=len(pinlist)
    for i in range(n):
        if (pinlist.beat_number[i]!=99):
          continue
        # i=0 i+=1 pinlist.dtypes
        pinlist.loc[i, 'cft_units_total'] = pinlist.loc[i, 'cft_units']
        pinlist.loc[i, 'units_total'] = pinlist.loc[i, 'units']
        # pinlist["cft_units_total"]=pinlist["cft_units"].copy(deep = True)     #pinlist["cft_units"]=pinlist["cft_units_total"]  
        # pinlist["units_total"]=pinlist["units"].copy(deep = True) 
        Van_ftl_units=((pinlist.loc[i,'units']*Van_ftl_cft)/pinlist.loc[i,'cft_units_total']).round(decimals=0)
        pincode_current=int(pinlist.CustomerPincode[i])
    
        # filtered_df = ist_by_pincode.loc[ist_by_pincode['pincode'] == pincode_current, 'avg ist']
        # filtered_df = ist_by_pincode.loc[ist_by_pincode['customer_pincode'] == str(pincode_current), '35th_percentile_ist']
        filtered_df = ist_by_pincode.loc[ist_by_pincode['customer_pincode'] == str(pincode_current), 'new_ist']
        IST_avg_mins = float(filtered_df.values[0]) if not filtered_df.empty else 10
        
        start_ist=pinlist.loc[i,'units']*IST_avg_mins
        start_stem_dist=pinlist.distance_hub[i]
        van_travel_time_hr=(start_stem_dist*2)/van_avg_speed
        # max_del_units=int((Van_oft_hrs_max-van_travel_time_hr)/(IST_avg_mins/60))         
        # max_del_cft=max_del_units/(pinlist.loc[i,'units']/pinlist.loc[i,'cft_units_total'])
        
        hub_ipp_cft=hub_ipp/(pinlist.loc[i,'units']/pinlist.loc[i,'cft_units_total'])
        van_oft_hrs=start_ist/60+van_travel_time_hr
        
        
        while pinlist.loc[i, 'units'] > hub_ipp or pinlist.loc[i, 'cft_units'] > Van_ftl_cft:
            vehicle_count += 1
            vehicle = [pincode_current]
            
            if pinlist.loc[i, 'cft_units'] > Van_ftl_cft:
                van_ftl_ist = Van_ftl_units * IST_avg_mins
                van_ftl_oft_hrs = van_ftl_ist / 60 + van_travel_time_hr
                pinlist.loc[i, 'cft_units'] -= Van_ftl_cft
                pinlist.loc[i, 'units'] -= Van_ftl_units
                vehicles.loc[len(vehicles)] = [vehicle_count, vehicle, Van_ftl_units, Van_ftl_cft, van_ftl_oft_hrs, van_ftl_ist / 60, van_travel_time_hr,van_travel_time_hr*van_avg_speed]
                    
            if pinlist.loc[i, 'units'] > hub_ipp:
                hub_ipp_ist = hub_ipp * IST_avg_mins
                hub_ipp_van_oft = hub_ipp_ist / 60 + van_travel_time_hr
                pinlist.loc[i, 'units'] -= hub_ipp
                pinlist.loc[i, 'cft_units'] -= hub_ipp_cft
                vehicles.loc[len(vehicles)] = [vehicle_count, vehicle, hub_ipp, hub_ipp_cft, hub_ipp_van_oft, hub_ipp_ist / 60, van_travel_time_hr,van_travel_time_hr*van_avg_speed]
            
        vehicle_count+=1
        vehicle=[pincode_current]
        pinlist.loc[i,'beat_number']=beat_number
        temp_pin_list=[]
        temp_list2=[]
        temp_pin_list.append(pincode_current)
        temp_list2.append(pincode_current)
        # start_load=pinlist.iloc[i]["units"] 
        start_load=pinlist.at[i,"units"]
        # start_cft=pinlist.iloc[i]["cft_units"]
        start_cft=pinlist.at[i,"cft_units"]
        load_counter=start_load
        Total_van_cft=start_cft
          
      
          
        start_ist=start_load*IST_avg_mins
        total_IST=start_ist
        # start_stem_dist=pinlist.distance_hub[i]
        # van_dist_counter=start_stem_dist
       
        # near_pincode_loc=i
        pin_add_flag=1
        
        points=[len(distance_mat_short)-1]
        for pin in temp_list2:
            points.append(pinlist[pinlist['CustomerPincode'] == pin].index[0])
        start_point = len(distance_mat_short)-1
        # shortest_path, shortest_distance = christofides_shortest_path(points, distance_mat_short, start_point)
        shortest_path, shortest_distance = new_christofides_shortest_path(points, distance_mat_short, start_point)
        # print("Shortest path:", shortest_path)
        # print("Shortest distance:", shortest_distance)
        
        van_dist_counter=shortest_distance
        van_oft_hrs=(start_ist/60)+(van_dist_counter/van_avg_speed)
        
        path_pincode=[]
        for y in shortest_path:
            if(y==len(distance_mat_short)-1):
              path_pincode.append('Hub')  
            else:
              path_pincode.append(pinlist.at[y,'CustomerPincode'])
            
              
        line="Beat:",beat_number,"Van_oft:",(start_ist/60)+(van_dist_counter/van_avg_speed),"total_IST:",start_ist/60,"van_travel_time_hr:",van_dist_counter/van_avg_speed,"From_Pincode:",pincode_current,"To_Pincode:","NA","Dist:","NA","Load:",pinlist.loc[i,'units'],"Total_van_cft:",pinlist.loc[i,'cft_units'],"vehicle_count:",vehicle_count
        # print(line)      
        line2=beat_number,(start_ist/60)+(van_dist_counter/van_avg_speed),start_ist/60,van_dist_counter,van_dist_counter/van_avg_speed,pinlist.loc[i,'units'],pinlist.loc[i,'cft_units'],path_pincode,temp_pin_list,vehicle_count
        output_list.append(line)
        output_list2.append(line2)
        
        while (pin_add_flag==1 and len(distance_df_var.columns)>1):  
           del_pin_loc=distance_df_var.columns.get_loc(pincode_current)
           distance_df_var=distance_df_var.drop(distance_df_var.columns[[del_pin_loc]], axis=1)
           temp_df=pd.DataFrame(columns=("pincode","nearbypincode","distance"))
           #find the nearby pincode for all pincode
           for pin in temp_pin_list:
               min_pin = distance_df_var.loc[pin].idxmin()
               min_pin_dis=distance_df_var.loc[pin].min()
               temp_df.loc[len(temp_df)] = [pin,min_pin,min_pin_dis]
               # temp_df.loc[len(temp_df)] = temp 
          
           # Find the row with the minimum distance
           min_row = temp_df.loc[temp_df['distance'].idxmin()]
           # Retrieve the necessary values
           min_d_pin = min_row['nearbypincode']
           min_d = min_row['distance']
           # Find the index of the pincode in pinlist
           pin_loc = pinlist[pinlist['CustomerPincode'] == min_d_pin].index.item()
           # Retrieve the corresponding values from pinlist
           near_pin_load = pinlist.at[pin_loc, 'units']
           near_pin_cft = pinlist.at[pin_loc, 'cft_units']
    
    
           points=[len(distance_mat_short)-1]    #add hub location for all routes
           
           temp_list2.append(min_d_pin)
           for pin in temp_list2:
               points.append(pinlist[pinlist['CustomerPincode'] == pin].index[0])
           # Subset of points for which shortest path is to be calculated
           start_point = len(distance_mat_short)-1
           shortest_path, shortest_distance = new_christofides_shortest_path(points, distance_mat_short, start_point)
           # print("Shortest path:", shortest_path)
           # print("Shortest distance:", shortest_distance)
           
           #assign the aggregated metrics
           load_counter=load_counter+near_pin_load
           
           Total_van_cft=Total_van_cft+near_pin_cft
           total_IST_old=total_IST  
           
           # filtered_df = ist_by_pincode.loc[ist_by_pincode['pincode'] == pincode_current, 'avg ist']
           filtered_df = ist_by_pincode.loc[ist_by_pincode['customer_pincode'] == str(pincode_current), 'new_ist']
           IST_avg_mins = float(filtered_df.values[0]) if not filtered_df.empty else 10
           
           total_IST=(total_IST+(IST_avg_mins*near_pin_load))
           
           van_dist_counter=shortest_distance
           van_travel_time_hr_old=van_travel_time_hr
           van_travel_time_hr=van_dist_counter/van_avg_speed
           van_oft_hrs_old=van_oft_hrs
           van_oft_hrs=total_IST/60+van_travel_time_hr
          
           path_pincode=[]
           for y in shortest_path:
               if(y==len(distance_mat_short)-1):
                 path_pincode.append('Hub')  
               else:
                 path_pincode.append(pinlist.at[y,'CustomerPincode'])
           
           # if(vehicle_count>0):
           #     vehicle_count+=1 
           line="Beat:",beat_number,"Van_oft:",van_oft_hrs,"total_IST:",total_IST,"van_travel_time_hr:",van_travel_time_hr,"From_Pincode:",pincode_current,"To_Pincode:",min_d_pin,"Dist:",min_d,"Load:",load_counter,"Total_van_cft:",Total_van_cft,"vehicle_count:",vehicle_count
           # line2=beat_number,van_oft_hrs,total_IST,van_dist_counter,van_travel_time_hr,load_counter,Total_van_cft,path_pincode
           # print(line)
        
           output_list.append(line) 
           # output_list2.append(line2)
        # This should change as per extended pincodes logic
        
           # if(min_d<=30 and Total_van_cft<=Van_ftl_cft and load_counter<=hub_ipp):   # total on field time cannot exceed 9hrs or inter pincode distance should not be more than 30 kms
           if(van_oft_hrs<=Van_oft_hrs_max and min_d<=30 and Total_van_cft<=Van_ftl_cft and van_dist_counter<=van_dist_max):   # total on field time cannot exceed 9hrs or inter pincode distance should not be more than 30 kms
           # # if(van_oft_hrs<=9 and min_d<=20):
               pin_add_flag=1
           else:
               pin_add_flag=0
        
        #assign the pincode to the existing beat
           if (pin_add_flag==1):
              # pinlist.beat_number[pin_loc]=beat_number    
              pinlist.loc[pin_loc,'beat_number']=beat_number 
              pincode_current=pinlist.loc[pin_loc,'CustomerPincode']   
              # near_pincode_loc=pin_loc.values[0]
              temp_pin_list.append(pincode_current)
              #remove pincode column from distance matrix
              line2=beat_number,van_oft_hrs,total_IST,van_dist_counter,van_travel_time_hr,load_counter,Total_van_cft,path_pincode,temp_pin_list,vehicle_count
              output_list2.append(line2)
              vehicle.append(pincode_current)
              van_oft_hrs_old=van_oft_hrs
           else:
              beat_number+=1
              # vehicles.append(vehicle)
              vehicles.loc[len(vehicles)] = [vehicle_count,vehicle,load_counter-near_pin_load,Total_van_cft-near_pin_cft,van_oft_hrs_old,total_IST_old/60,van_travel_time_hr_old,van_travel_time_hr_old*van_avg_speed]
       
        if (len(distance_df_var.columns)<=1 and pin_add_flag==1):
          vehicles.loc[len(vehicles)] = [vehicle_count,vehicle,load_counter,Total_van_cft,van_oft_hrs,total_IST/60,van_travel_time_hr,van_travel_time_hr*van_avg_speed]
   
    # 
    pinlist['cft_units_total'] = pinlist['cft_units_total'].fillna(pinlist['cft_units'])
    pinlist['units_total'] = pinlist['units_total'].fillna(pinlist['units'])
    pinlist = pinlist.drop(['cft_units', 'units'], axis=1)
    pinlist['Attainment'] = pd.to_numeric(pinlist['Attainment'])
    pinlist['units_total'] = pd.to_numeric(pinlist['units_total'])
    
    # Calculate weighted attainment for each beat number
    # weighted_attainment = pinlist.groupby('beat_number').apply(lambda x: (x['Attainment'] * x['units_total']).sum() / x['units_total'].sum()).reset_index(name='weighted_attainment')
    # weighted_attainment="NA"
    pinlist.to_csv('op/pinlist/'+hub+'_pinlist.csv', index=False)
    
    # # Print the allocated vehicles
    # print("Number of vehicles needed:", len(vehicles))
    # for i, vehicle in enumerate(vehicles):
    #     print(f"Vehicle {i + 1}: {vehicle}")   
    # vehicles.round(decimals=1, inplace=True)
    # Merge the weighted_attainment DataFrame with other_df based on 'beat_number'

    vehicles[['Shipments','Total_van_cft','van_travel_time','van_oft_hrs','IST','van_travel_dist']] = vehicles[['Shipments','Total_van_cft','van_travel_time','van_oft_hrs','IST','van_travel_dist']].round(1)
    # vehicles['van_oft_hrs'] = vehicles['van_oft_hrs'].round(1)
    vehicles["beat_index"]=np.where(vehicles['van_travel_dist']<=120,np.where(vehicles['Shipments']>=27,1,vehicles['Shipments']/35),np.where(vehicles['Shipments']>=35,1,vehicles['Shipments']/35))
    vehicles["lm_beat_index"]=(1/vehicles["beat_index"]).round(decimals=1)
    vehicles.rename(columns={'vehicle_count': 'beat_number'}, inplace=True)
    
    # Merge Weighted assigment here
    # vehicles = pd.merge(vehicles, weighted_attainment, on='beat_number', how='left')
    
    # vehicles = pd.merge(vehicles, hubwise_dailybeat, on='CurrentHubName', how='left')
    # vehicles_daily=vehicles[vehicles['Shipments']<20 | ]
    # unique_pincodes = vehicles['pincodes'].explode().unique()
    # new_pin = pd.DataFrame({'pincode': unique_pincodes})
    
    vehicles.to_csv('op/hub_beat_op/'+'hubbeat_'+hub+'.csv', index=False)
    
   
    
    # # add pincodes and beats on a map
    
    import folium
    pinlist['beat_color']=np.where(pinlist["beat_number"] == 1,'red',np.where(pinlist["beat_number"] == 2,'blue',np.where(pinlist["beat_number"] == 3,'green',
                                    np.where(pinlist["beat_number"] == 4,'pink',np.where(pinlist["beat_number"] == 5,'orange',np.where(pinlist["beat_number"] == 6,'darkgreen',np.where(pinlist["beat_number"] == 7,'darkblue',
                                    np.where(pinlist["beat_number"] == 8,'gray',np.where(pinlist["beat_number"] == 9,'purple','lightgray')))))))))
    
    
    map_new=folium.Map(location=[pinlist['hub_latitude'].values[0].round(decimals=1),pinlist['hub_longitude'].values[0].round(decimals=1)],zoom_start=10)
    list_coor=pinlist[['CustomerPincode','pincode_latitude','pincode_longitude','units_total','cft_units_total','beat_number','beat_color']].values.tolist()
    
    # i=0 i+=1
    for i in list_coor:
        map_new.add_child(folium.Marker(location=[i[1],i[2]],
                                    popup=i[4],tooltip=i[0],icon=folium.Icon(color=i[6])))
    
    
    map_new.add_child(folium.Marker(location=[pinlist['hub_latitude'][0],pinlist['hub_longitude'][0]],icon=folium.Icon(color="black")))
    map_new.save('op/map/'+hub+'_map_view.html')
    
    return pinlist,vehicles
    
    # beat_summary=pd.DataFrame(output_list2,columns=['beat_number', 'van_oft_hrs' , 'total_IST_hrs', 'van_dist_kms', 'van_travel_time_hr', 'total_units', 'Total_van_cft', 'shortest_path','pincode_list','vehicle_count'])
    # # beat_summary=beat_summary[['beat_number','shortest_path','pincode_list','vehicle_count']]
    # beat_summary.drop(columns=['total_units', 'Total_van_cft'], axis=1, inplace=True)
    
    # # pin_count_beat_summary=beat_summary.groupby("beat_number")['shortest_path'].count().rename('bs_count').reset_index() 
    # # beat_summary=pd.merge(beat_summary, pin_count_beat_summary[['beat_number','bs_count']], how="left", on="beat_number")
    # # beat_summary=beat_summary.loc[((beat_summary.van_oft_hrs<=9) & (beat_summary.Total_van_cft<=Van_ftl_cft)) | (beat_summary.bs_count==1)].round(decimals=2)
    # beat_summary=beat_summary.groupby("beat_number").last().reset_index().round(decimals=2)
    
    # # pin_count=pinlist.groupby("beat_number")['CustomerPincode'].count().reset_index()
    # # pin_count.columns = ['beat_number', 'pin_count']
    # # beat_summary=pd.merge(beat_summary, pin_count[['beat_number','pin_count']], how="left", on="beat_number")
    
    # pinlist['cft_units_total'] = pinlist['cft_units_total'].fillna(pinlist['cft_units'])
    # pinlist['units_total'] = pinlist['units_total'].fillna(pinlist['units'])
    # pinlist = pinlist.drop(['cft_units', 'units'], axis=1)
    
    
    # agg_df = pinlist.groupby('beat_number').agg(cft_units_total=('cft_units_total', 'sum'), units_total=('units_total', 'sum'),fdd_cft_units=('fdd_cft_units', 'sum'), fdd_units=('fdd_units', 'sum'), pin_count=('CustomerPincode', 'count')).reset_index()
    # beat_summary = pd.merge(beat_summary, agg_df[['beat_number', 'cft_units_total', 'units_total', 'fdd_cft_units', 'fdd_units','pin_count']], on='beat_number').drop_duplicates(subset='beat_number')
    # # total_cft=pinlist.groupby("beat_number")['cft_units_total'].sum().rename('cft_units_total').reset_index() 
    # # beat_summary=pd.merge(beat_summary, total_cft[['beat_number','cft_units_total']], how="left", on="beat_number")
    # # beat_summary['van_utl']=beat_summary['Total_van_cft']/Van_ftl_cft
    # beat_summary['van_utl'] = (beat_summary['cft_units_total'] / ((Van_ftl_cft/cft_factor)*beat_summary['vehicle_count'])) * 100
   
    # pinlist=pd.merge(pinlist, beat_summary[['beat_number','van_utl']], how="left", on="beat_number")
    # pinlist['daily_flag']=np.where(pinlist['van_utl']>=70,1,0)
    # pinlist = pd.merge(pinlist, ds_pbh_map[['pincode','beat','beat_index']], how="left",left_on="CustomerPincode", right_on="pincode")
    
    # beat_summary['van_utl'] = beat_summary['van_utl'].map('{:.2f}%'.format)
    # # import math
    # # pinlist['beat_freq'] = pinlist['van_utl'].apply(lambda x: math.floor(100 / x))
    # pinlist_raw.to_csv('op/'+ hub+'_pinlist_raw.csv', index=False)
    # df=pd.DataFrame(output_list)
    # df.to_csv('op/model_op/'+hub+'_model_ouput.csv', index=False)
    # beat_summary.to_csv('op/beat_summary/'+hub+report_run_date.strftime('%Y-%m-%d')+'_beat_summary.csv', index=False)
    # # pinlist_raw.to_csv('op/'+ hub+'_pinlist_raw.csv', index=False)
    # # pinlist.to_csv('op/pinlist/'+hub+'_pinlist.csv', index=False)
    # pinlist.to_csv('op/pinlist/'+hub+'_pinlist'+report_run_date.strftime('%Y-%m-%d')+'.csv', index=False)
    # # distance_df.to_csv('op/'+hub+'_distance_matrix.csv', index=False)
    
    

   

Hub_wise_beat_summary=pd.DataFrame(columns=['hub_name','beat_index','daily_beat','shipments','old_dailybeat'])
# beat_final=pd.DataFrame()    
# for hub in hub_list:
#     print(hub)
#     pinlist,vehicles=run_model(hub,input_file)
#     hub_beat_index=(((vehicles['Shipments'] * vehicles['lm_beat_index']).sum())/vehicles['Shipments'].sum()).round(decimals=1)
#     daily_beat = (vehicles.loc[vehicles['lm_beat_index'] == 1, 'Shipments'].sum()/vehicles['Shipments'].sum()).round(decimals=2)
#     Hub_wise_beat_summary.loc[len(Hub_wise_beat_summary)]= [hub,hub_beat_index,daily_beat,vehicles['Shipments'].sum()]
  
for hub in hub_list:
    try:
        # print(hub)
        pinlist, vehicles = run_model(hub, input_file)
        hub_beat_index = (((vehicles['Shipments'] * vehicles['lm_beat_index']).sum()) / vehicles['Shipments'].sum()).round(decimals=1)
        daily_beat = (vehicles.loc[vehicles['lm_beat_index'] == 1, 'Shipments'].sum() / vehicles['Shipments'].sum()).round(decimals=2)
        old_dailybeat = (pinlist.loc[pinlist['daily_beat_flag'] == 1, 'units_total'].sum() / pinlist['units_total'].sum()).round(decimals=2)
        # hubwise_old_dailybeat = pinlist.groupby('CurrentHubName').apply(lambda x: (x['daily_beat_flag'] * x['units_total']).sum() / x['units_total'].sum()).reset_index(name='old_dailybeat')
        # old_dailybeat=pd.to_numeric(hubwise_old_dailybeat["old_dailybeat"])
        Hub_wise_beat_summary.loc[len(Hub_wise_beat_summary)] = [hub, hub_beat_index, daily_beat, vehicles['Shipments'].sum(),old_dailybeat]
       
    except Exception as e:
        if "divide by zero" in str(e):
            print(f"RuntimeWarning for hub {hub}: {e}")
            warnings.warn(f"RuntimeWarning for hub {hub}: {e}", RuntimeWarning)
        else:
            print(f"Error processing {hub}: {e}")
        continue    
    
# for hub in hub_list:
#     try:
#         # print(hub)
#         pinlist, vehicles = run_model(hub, input_file)
#         hub_beat_index = (((vehicles['Shipments'] * vehicles['lm_beat_index']).sum()) / vehicles['Shipments'].sum()).round(decimals=1)
#         daily_beat = (vehicles.loc[vehicles['lm_beat_index'] == 1, 'Shipments'].sum() / vehicles['Shipments'].sum()).round(decimals=2)
#         Hub_wise_beat_summary.loc[len(Hub_wise_beat_summary)] = [hub, hub_beat_index, daily_beat, vehicles['Shipments'].sum()]
#     except RuntimeWarning as rw:
#         print(f"RuntimeWarning for hub {hub}: {rw}")
#         continue
#     except Exception as e:
#         print(f"Error processing {hub}: {e}")
#         continue    
       

Hub_wise_beat_summary.to_csv('op/Hub_wise_beat_summary.csv', index=False)    
# dfs.append(pinlist_final[['CurrentHubName','CustomerPincode','daily_flag']])
# beat_final.append(beat_summary)
# pinlist_final.to_csv('op/'+'pinlist_final_'+report_run_date.strftime('%Y-%m-%d')+'.csv', index=False)    


# distance_df_var.to_csv('op/'+'dist_mat.csv', index=False)    
    
# merged_df = pd.concat([df.set_index(['CurrentHubName','CustomerPincode']) for df in dfs], axis=1).reset_index()

# original_columns = merged_df.columns.tolist()
# new_columns = [original_columns[0], original_columns[1]] + [f"{col}_day_{i+1}" for i, col in enumerate(original_columns[2:])]

# # Rename the columns in the merged DataFrame
# merged_df.columns = new_columns  
# merged_df.to_csv('op/'+'pinlist_merged.csv', index=False)    