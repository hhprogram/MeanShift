import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


# covers videos #41 -42 of sentdex machine learning with Python tutorials. Note: used his 
# code as guide but tried to code up my own version

# basically makes all points clusters at first. Then for each cluster we look at increasingly larger
# radiuses. We then weight data points closer to a cluster higher than ones lying farther away from
# a cluster. We then calculate new clusters with these weightings. As we do this some of the
# original clusters will get very close to each other / overlap which then we can consolidate and
# decrease the number of clusters. We keep doing this until the number of clusters stabilizes AND 
# all clusters move less than the given threshold. then we have all our clusters and their locations
# then to predict we just take an input vector and classify it into the group with whichever cluster
# point the input vector is closest to

# other general notes: Seems that a self.radius that is too small tends to result in too many
# centroids and a too large radius results in too few usually

class Mean_Shift():
    def __init__(self, radius=None, radius_step=100):
        # - radius_step used to denote how many intervals we want to have when stepping through when
        # calculating clusters. we do all radius_steps when calculating new clusters. And just 
        # weight data points according to which radius they fall within. So cluster locations get
        # updated every time we've gone through all RADIUS_STEP number of steps. 

        # - weights is a list that will just hold numbers to be used to
        # 'weight' data values found within certain radiuses of a cluster. We reverse it as we'll 
        # just we walking through the weights list from index 0 to the last index and thus we want
        # the first weights to be the highests as those will represent the datapoints closest to 
        # the cluster centroid. do range from 1 - RADIUS_STEP inclusive as if i just
        # did range(self.radius_step) then one weight value would be zero and that could
        # cause issues (as it did initially with 0 division)

        # - self.threshold is a threshold number that will be used to determine if certain cluster
        # centroids have converged to one or if we have optimized as far as we can go

        # - self.max_iter is the max number of iterations we will run even if the centroids haven't
        # been 'optimized' to avoid very very long run times

        # - self.final_clusters to be what the clusters are after we run FIT method. Will be 
        # reassigned everytime we run fit() (will be a list of tuples - (label, location))

        # - cluster_mapping is a dict to hold the final mapping of training data points to each 
        # cluster point. Mostly for ease of analysis / visualization (reassigned with each call
        # of fit) Key is the 'label' of the cluster and the value is a
        # list of all training data points belonging to that cluster
        self.radius = radius
        self.radius_step = radius_step
        self.weights = [i*100 for i in range(self.radius_step)]
        self.weights.reverse()
        self.threshold = .0000000001
        self.max_iter = 600
        self.final_clusters = []
        self.cluster_mapping = {}

    def fit(self, data):
        """
        'fits' this dynamic radius/bandwith K Mean_Shift algorithm. It goes through a bunch of
        radiuses around datapoints to try to figure out the best distribution/number/location of
        all our clusters. Also, Mean_Shift chooses its own number of 'k' clusters that it thinks
        groups the data best
        """
        # if no radius given then we will assume we want a dynamically set radius which will be used
        # to set the number of clusters dynamically
        if not self.radius:
            all_data_avg = np.average(data)
            # note the below gets the norm between the vector all_data_avg and the origin. This norm 
            # will be used as the biggest possible radius (ie an estimate for 'range' of our data
            # as we need stop expanding our radius at some point).
            # Note, this could cause some trouble with 
            # pathological data (ex. if have 4 clusters and each one evenly spread out in each
            # quadrant and thus causing the all_data_avg to be just as the origin and then
            # the radius would be zero and no matter how many radius steps there are
            # would never be any other nodes in any cluster since the search radius is 
            # zero). We make the initial radius the norm divided by the step as we want the 
            # initial radius to be small and then we will make it larger and larger but with each
            # increase of the radius - we will weight the points within that radius lower and thus
            # be less influential in creating a cluster. Therefore the last radius circle will be 
            # just self.radius * self.radius_step which will just equal to the all_data_norm
            all_data_norm = np.linalg.norm(all_data_avg)
            self.radius = all_data_norm / self.radius_step

        # a dict that stores the clusters. initialize each training data point to be a cluster at 
        # first.then key is the 'label' values is a list (1) list of weighted vectors (2) number of 
        # vectors (3) the actual location of cluster. We don't have the actual training data points
        # that are classified as 'label' in this dictionary. Will do it at the end of the fit 
        # method (will be just like we are 'predicting' for each training data point) (4) is the
        # location of this cluster's previous centroid (which will be used when seeing if we
            # have stabilized and thus are optimized and can stop)
        # side note: changed this to a list because literal values in a tuple are immutable. Thus
        # when i'd try to update the value at index 1 it wouldn't allow me to.( throw an error)
        # initialize it with just itself as the weighted vectors and 1. Do this as could be possible
        # a point is all by itself and then if no points within radius then have division by 0 issue
        clusters = {}
        for label, point in enumerate(data):
            clusters[label] = [[point], 1, point, None]
        loops = 0
        prev_radius = 0
        # print(clusters)
        # do this instead of a while loop to avoid infinite loops
        for _ in range(self.max_iter):
            # loop through each exiting cluster centroid and calc distances between it and all
            # datapts
            for label in clusters:
                for point in data:
                    distance = np.linalg.norm(point - clusters[label][2])
                    # print(distance)
                    # the way we use our weight list is to see how many 'radiuses' away this certain
                    # point is from the cluster centroid. The fewer mulitples of radiuses away then
                    # the smaller int(distance / self.radius) will be and thus we willl pick a
                    # weight number earlier on in the weight list which is a bigger number. We also
                    # cap it at self.radius_step-1 as our weight list is only self.radius_step #
                    # of elements long. Thus super far away vectors get the smallest weight
                    weight_index = min(int(distance / self.radius), self.radius_step-1)
                    # print(weight_index)
                    # then we do this vs. his method of putting Weight number of duplicates and then
                    # averaging that iterable by just scaling the vector by weight and then adding
                    # weight to the denominator to avoid creating very large lists
                    weight = self.weights[weight_index]
                    clusters[label][0] += weight*point
                    clusters[label][1] += weight


            # after doing all bandwidths we then need to calculate new clusters and then compare
            # their locations to their old locations
            # use this to compare to old cluster locations and see if they keep moving. If all of 
            # them have moved less than SELF.THRESHOLD then we can stop. Also use to repopulate 
            prev_clusters = dict(clusters)
            # reset the dictionary as this is the latest clusters dictionary
            clusters = {}
            # first recalculate cluster locations
            for label in prev_clusters:
                cluster_info = prev_clusters[label]
                # new cluster is the weighted average of the data points that fell within range
                # of the older cluster points. The weights are just squared value of which 
                # smallest 'radius circle' the datapoint is within to the old cluster. thus
                # data point that is within a smaller radius of a cluster point is weighted 
                # higher. (hoping calc the avg vector like this is more efficient than 
                # his method of putting WEIGHT copies of each vector into an iterable
                # then taking the average of vector over all those vectors. Basically just
                # a lot of duplicates of vectors in highly weight radiuses and making
                # such large iterables is probably expensive)
                # doing on axis=0 means it will result in one vector whose values for each index
                # will be the sume of that index over all the vectors. If default no axis arg 
                # given then just will return a scalar value that is sum of all values axis=1
                # returns a vector that has 'n' elements where 'n' is the number of arrays to sum
                # over and each element in this vector is just the sum of the ith vector values
                if prev_clusters[label][1] == 0:
                    print("odd", prev_clusters[label])
                new_cluster = np.sum(cluster_info[0], axis=0) / cluster_info[1]
                # print("new:", len(cluster_info[0]))
                old_cluster = prev_clusters[label][2]
                clusters[label] = [[new_cluster], 1, new_cluster, old_cluster]

            # loop through new clusters dictionary and see if any clusters are within the threshold
            # of each other we consolidate them. not trying to optimize the number of clusters to
            # consolidate / which ones to pick becauase I think that is a set cover problem which
            # is NP complete. So just loop through the new clusters and put in a label to be 
            # consolidated when we encounter a pair closer to each other than the THRESHOLD. 
            # make sure we don't add both clusters of a pair by checking and seeing if the other
            # cluster is already in the 'to be removed' list then don't add the cluster
            to_remove = set()
            for label in clusters:
                for label2 in clusters:
                    # we do nothing if we are just comparing the exact same cluster to each other
                    if not label == label2:
                        label_location = clusters[label][2]
                        label2_location = clusters[label2][2]
                        if np.linalg.norm(label_location - label2_location) < self.radius:
                            # add label iff label2 isn't already in the set to_remove. This shall 
                            # prevent over deletion of clusters as if didn't have this check then
                            # both clusters of a pair that were within self.threshold of each other
                            # would be deleted leaving no cluster when in reality there should be
                            # one that now contains the training data points of the original pair
                            # since it is a set we can just keep 'adding' the same label to
                            # to_remove without worrying that we'd try to delete label multiple 
                            # times and get an error
                            if label2 not in to_remove:
                                to_remove.add(label)

            for label in to_remove:
                clusters.pop(label, None)

            # locations = [clusters[location][2] for location in clusters]
            # print(loops, locations)
            # then we loop through to calc the distance. For efficiency we break out of loop if 
            # one of them is above threshold. Boolean to be used to tell if we should also break 
            # out of our outer most for loop that just keeps going until we reach self.max_iter
            optimized = True
            for label in clusters:
                new_cluster = clusters[label][2]
                old_cluster = clusters[label][3]
                if np.linalg.norm(new_cluster - old_cluster) > self.threshold:
                    optimized = False
                    break
            # not strictly necessary 4 lines of code but we do this to keep label values consecutive
            # and as small as possible. Basically, there will be labels after we have 'merged'
            # some clusters together that are no longer in order. This just re labels the remaining
            # labels so that it goes from zero to (# of labels-1) in consecutive order
            prev_clusters = dict(clusters)
            clusters = {}
            for new_label, label in enumerate(prev_clusters):
                clusters[new_label] = prev_clusters[label]

            if optimized:
                print("optimized")
                break 
            loops+=1
        # make the clusters list have the location of cluster and a class 'label'. Remember label
        # is randomly assigned so doesn't really mean anything just used for categorizing datapoints
        # together. Put class first because for simplicity sake I could sort on class if i wanted
        # to. Which i will be doing as in theory by the end of this algorithm should have relatively
        # few clusters in this list and therefore sorting it should be extremely fast. So I am going
        # to sort the list (and it will sort by first element of the tuple only - by default), and
        # then return the index of the min distance and because I have sorted by 'label' this index
        # will match with the 'label'
        self.final_clusters = [(label, clusters[label][2]) for label in clusters]
        self.final_clusters.sort()
        # final loop to then put all the training DATA with their clusters just for ease of graphing
        # purposes etc..
        print(loops)
        for point in data:
            # list of distances from each label. We first go through all the labels by going through
            # our sorted self.final_clusters list. then use that label value to find in our 
            # clusters dict the actual location of that label's centroid
            distances = [np.linalg.norm(cluster[1] - point) for cluster in self.final_clusters]
            # note this means if there is a tie then it just picks the cluster with the lowest
            # 'label' value (ie comes first in the list). So this gets the index of the smallest 
            # distance and then i use this index to reference self.final_clusters (remember index
            # = label) and I do this because self.final_clusters has the cluster's actual 
            # location which i also need
            closest_cluster = self.final_clusters[distances.index(min(distances))][0]
            if closest_cluster not in self.cluster_mapping:
                self.cluster_mapping[closest_cluster] = [point]
            else:
                self.cluster_mapping[closest_cluster].append(point)

        percent_clusters = len(self.final_clusters) / len(data)

        if percent_clusters > .35:
            self.refit(data, percent_clusters)

    def refit(self, data, percent, new_radius=.08):
        """
        helper method to re run fit if we have found way too many centroids for the data. In theory
        could run if found way too few but that might be harder to check whether or not we have 
        too few centroids
        """
        print("ran a refit because percent centroids was: {}%".format(percent*100))
        print("Old radius was {} and now we are switching to {}".format(self.radius, new_radius))
        self.radius = new_radius
        self.fit(data)


    def predict(self, data_pt):
        # just takes a single DATA_PT and compares it to the distances of all the cluster centroids
        # and returns the label whose centroid is closest to DATA_PT
        distances = [np.linalg.norm(cluster[1] - data_pt) for cluster in self.final_clusters]
        return self.final_clusters[distances.index(min(distances))][0]

# make blobs makes random generated blobs of data to test clustering. first value in the tuple
# it returns is a list of all the input data points. And the 2nd (y) is a list of the cluster label
# ground truth values (can use to test how our meanshift is doing)

X, y = make_blobs(n_samples=35, centers=4, n_features=2)

# this data set below was making my algorithm create way too many centroids. almost one for each 
# datapoint. If I increased the self.radius to .08 it helped. It was because the self.radius that 
# was set was way too small thus there were a lot of amost overlapping centroids

# X= np.array([[-8.07075573,0.23228501],
#  [-1.13012621,-7.25059682],
#  [1.38747373,-1.72230525],
#  [-9.78197325,-3.18955079],
#  [-9.01995239,-1.02928903],
#  [-1.38865489,-4.72598967],
#  [1.68060129,5.1969551 ],
#  [2.28232516,5.93508202],
#  [0.80198814,-2.81074344],
#  [2.90583616,-1.26001706],
#  [-0.91872464,-5.70456067],
#  [1.31873379,5.68368078],
#  [1.98719292,-3.50287446],
#  [1.61509236,4.4850154 ],
#  [-1.12533547,-4.38729602],])

# this dataset was only giving me one centroid when I added the line: 
# self.radius = max(all_data_norm / self.radius_step, .08). If i just went back to normal
# self.radius = all_data_norm / self.radius_step...then my algorithm performed well

# X=np.array([[-1.16969289,-2.49928124],
#  [-3.85262601,-9.58263141],
#  [-0.11671738,-1.8191964],
#  [-3.90866867,3.26052815],
#  [-3.97737999,1.85978679],
#  [-3.68274047,4.60431449],
#  [-6.39726654,-10.40068199],
#  [-7.71046233,-2.44189925],
#  [-9.1424346,-2.67972543],
#  [-3.70557342,2.13695899],
#  [-3.97305218,-9.49966754],
#  [-3.78081496,-7.85522323],
#  [-1.4557809,-0.70602581],
#  [-0.46225276,-2.02102747],
#  [-9.15017039,-2.77286816]])


# dataset that was resulting in way too many centroids. then i added the refit method
# X=np.array([[  0.40940364,  -2.72105654],
#  [  6.8589362,   -1.78694798],
#  [ -8.80996269,   5.15545812],
#  [  1.93572018,  -5.1557829 ],
#  [  9.6366184,   -2.0512005 ],
#  [ -3.07363483,  -4.04830755],
#  [  1.39882403,  -1.02606161],
#  [ -1.10468197,  -3.31456745],
#  [  7.34143997,  -0.86269288],
#  [  6.20996616,  -2.03134141],
#  [ -9.06614402,   4.45690799],
#  [  0.96304246,  -4.60945352],
#  [ -9.68480132,   4.72113316],
#  [  2.10288004,  -3.43693049],
#  [  0.59247274,  -4.85518105],
#  [  8.48477037,  -0.75349998],
#  [ -1.59009353,  -3.58970874],
#  [ -2.14917964,  -5.40053638],
#  [-10.38650375,   2.88252302],
#  [  7.95491554,  -2.44617339],
#  [ -1.00971004,  -5.35683182],
#  [  1.63795198,  -2.8641408 ],
#  [ -8.07356037,   4.24452855],
#  [ -8.76536591,   4.91854764],
#  [ -2.30128027,  -5.61323095],
#  [  6.24834952,  -2.52966231],
#  [ -7.41929501,   4.58948366],
#  [  1.38458426,  -2.77429755],
#  [ -1.78613505,  -3.19167054],
#  [  2.21986522,  -3.04794981],
#  [ -7.66672095,   3.56107806],
#  [  2.79890484,  -4.43574076],
#  [ -9.33057086,   4.26841643],
#  [  8.37066998,  -3.60293953],
#  [  7.99371417,  -1.23643485]])


##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k", "y", "m"]

clf = Mean_Shift()
clf.fit(X)

centroids = clf.final_clusters
print("radius:", clf.radius)
print(X)
print("# of centroids:", len(centroids))
print(centroids)
# plot the centroids based on what it was trained on X
for centroid in centroids:
    label = centroid[0]
    color = colors[label]
    plt.scatter(centroid[1][0], centroid[1][1], color=color, marker='*', s=150, linewidths=5)
    [plt.scatter(pt[0], pt[1], color=color, marker='o', s=100, linewidths=5, alpha=.3) for pt in clf.cluster_mapping[label]]

# for num, pt in enumerate(X):
#     color = colors[y[num]]
#     plt.scatter(pt[0], pt[1], color=color, marker='o', s=150, linewidths=5, alpha=.5)

# for label in clf.cluster_mapping:
#     color = colors[label]
#     pts = clf.cluster_mapping[label]
#     [plt.scatter(pt[0], pt[1], color=color, marker='*', s=150, linewidths=5) for pt in pts]

plt.show()


