import numpy as np
import random
from connect import *
import sys
import time

##calculate distEuclid
def distEuclid(x,y):
    return np.sqrt(np.sum((x-y)**2))


##start define KMeans cluster
#kmeans++
def KMeans(data, k, is_random = False):
    num,dim = data.shape #get the number of points and dimension
    # get a random element from 0 to num - 1 as the index of the first centerpoint
    first_idx = random.sample(range(num), 1)    # eg. first_index = 3
    centerpoint = data[first_idx] #get the first centerpoint  # eg. centerpoint = data[3]
    dist_note = np.zeros(num) #[0, 0, ..., 0]
    dist_note += 100 #[100, 100, ..., 100]

    # loop from 0 to k - 1
    # select k centerpoints
    for j in range(k):
        #calculate every dist, save minimum distance
        # loop through all the points
        for i in range(num):
            dist = distEuclid(centerpoint[j], data[i])
            # update dist_node[i] to dist if dist < dist_note[i]
            if dist <  dist_note[i]:
                dist_note[i] = dist 
        #use roulette wheel selection  
        if is_random:
            #eg. [1, 2, 3, 4] / 10 = [0.1, 0.2, 0.3, 0.4]
            # add up to 1, because it is a probability distribution
            dist_p = dist_note / dist_note.sum()
            # randomly select an index from 0 to num - 1 based on the probability distribution
            # the point with a larger distance from center point has a higher probability of being selected
            next_idx = np.random.choice(range(num), 1, p=dist_p)
            # add the selected point to the centerpoint
            centerpoint = np.vstack([centerpoint, data[next_idx]])
        else:
            # select the farthest point as the next centerpoint
            next_idx = dist_note.argmax()
            centerpoint = np.vstack([centerpoint, data[next_idx]])
    
    ##record Information 0:belong to which cluster 1:distance to centerpoint of cluster 
    #initiate cluster of shape (num,2)
    cluster = np.zeros((num,2))
    #initiate the 0th column of points to -1
    cluster[:,0]=-1
    ##record if points' cluster change
    change = True
    ##initiate centroids
    count = 0
    cp = centerpoint
   
    while change:
        count += 1  
        # update the points' cluster for at most 100 times
        if count > 100:
            break
        #loop through every point to find the nearest centerpoint to each point
        for i in range(num):
            # set the minimum distance to a large number
            minDist = 9999.9
            # set the index of the nearest centerpoint to -1
            minIndex = -1
            
            ##Calculate distance between points and centerpoints, find the nearest centerpoint
            for j in range(k):
                dis = distEuclid(cp[j],data[i])
                # update the minDist and minIndex if the distance is smaller
                if dis < minDist:
                    minDist = dis
                    minIndex = j
                    
            ##if the cp is not belonging to this cluster, change the cluster
            if cluster[i,0]!=minIndex:
                change = True
            # update the cluster of the point
            cluster[i,:] = minIndex,minDist
        
        ##   Calculate new centerpoint
        for j in range(k):
            # get all the points in the cluster
            pointincluster = data[[x for x in range(num) if cluster[x,0]==j]]
            # update the centerpoint of the cluster to the mean of all the points in the cluster
            cp[j] = np.mean(pointincluster,axis=0)
            
    ##record the points in each cluster
    ptv_clusters = {i: [] for i in range(k)}
        
    # loop through all the points
    # add the point to the corresponding cluster
    for i in range(len(data)):
        # get the index of the cluster of this point
        cluster_index = int(cluster[i, 0])
        # add the point to the corresponding cluster
        ptv_clusters[cluster_index].append(roi_names[i])  # Use roi_names list
       
    # return the centerpoints and the cluster of points
    return cp, cluster, ptv_clusters



# decide k 
# cp: centerpoints of clusters
# cluster: cluster of points
def if_reach_requirement(cp, cluster):
    # loop through all the centerpoints
    for c in range(len(cp)):
        d = data[cluster[:,0] == c]
        for i in range(len(d)):
            dist = distEuclid(cp[c], d[i])
            if dist > 4.5:
                return False
    return True


# Set the random seed to 42
random.seed(42)
# the maximum number of points
num = 50 #points    
case = get_current("Case")
plan = get_current("Plan")
examination = get_current("Examination")
colors = np.array(['Yellow', 'Red', 'Green', 'Orange', 'Blue', 'Pink', 'Cyan','Yellow', 'Red', 'Green', 'Orange', 'Blue', 'Pink', 'Cyan','Yellow', 'Red', 'Green', 'Orange', 'Blue', 'Pink', 'Cyan','Yellow', 'Red', 'Green', 'Orange', 'Blue', 'Pink', 'Cyan'])

# Get the center of the PTV ROI for the structure set of the plan.
structure_set = case.PatientModel.StructureSets[0]

# Create an empty list to store the names of the PTV ROIs
roi_names = []
# Use arr1 to store the coordinates of the PTVs
arr1 = []
for bb in structure_set.RoiGeometries:
    roi_name = bb.OfRoi.Name
    nme=str(roi_name)
    
    if 'PTV' in nme: 
     if 'Middle' not in nme: 
      if 'Outer' not in nme: 
       if 'Inner' not in nme:
           ptv_center = structure_set.RoiGeometries[roi_name].GetCenterOfRoi()
           #print(ptv_center.x, ptv_center.y, ptv_center.z)
           arr1 = np.append(arr1, [ptv_center.x, ptv_center.y, ptv_center.z])
           #print(roi_name)
           roi_names.append(roi_name)  # Add the ROI name to the list

 
 

# make sure data is in the shape of (num, 3)
data = arr1.reshape(-1,3)


# loop through all the possible k values from 1 to 20
# find the smallest k that satisfies the requirement
for k in range(1, 21):
    cp, cluster, ptv_clusters = KMeans(data,k)
    #print(cp, data)
    #print(len(cp))
    result = if_reach_requirement(cp, cluster)
    if result == True:
        break

# get the centerpoints and the cluster of points
cp, cluster, ptv_clusters = KMeans(data,k)

#print(ptv_clusters)
# loop through all the centerpoints
for i in range(0, len(cp)):
    # get all the PTVs in the cluster
    ptvs_in_cluster = ptv_clusters[i]
    ptv_numbers = []
    for ptv in ptvs_in_cluster:
        # Extract all numeric values from the PTV name
        print(ptv)
        # get all the numeric parts of the PTV name
        numeric_parts = [part for part in ptv.split() if part.isdigit()]
        # add the numeric parts to the list of PTV numbers
        ptv_numbers.extend(numeric_parts)
    # create a iso name
    iso_name = f"ISO_{','.join(ptv_numbers)}"
   
    case.PatientModel.CreatePoi(Examination=examination, Point={'x': cp[i, 0], 'y': cp[i, 1], 'z': cp[i, 2]}, Name=iso_name, Color=str(colors[i]), VisualizationDiameter=0.5, Type="Isocenter")
