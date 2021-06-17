import dijkstra3d
import numpy as np

field = np.ones((512, 512, 512), dtype=np.int32)
source = (0,0,0)
target = (5, 5, 5)

# path is an [N,3] numpy array i.e. a list of x,y,z coordinates

# terminates early, default is 26 connected

path = dijkstra3d.dijkstra(field, source, target, connectivity=6) 

#path = dijkstra3d.dijkstra(field, source, target, bidirectional=True) # 2x memory usage, faster
#
print(path)

# Use distance from target as a heuristic (A* search)
# Does nothing if bidirectional=True (it's just not implemented)


# path = dijkstra3d.dijkstra(field, source, target, compass=True) 

# parental_field is a performance optimization on dijkstra for when you
# want to return many target paths from a single source instead of
# a single path to a single target. `parents` is a field of parent voxels
# which can then be rapidly traversed to yield a path from the source. 
# The initial run is slower as we cannot stop early when a target is found
# but computing many paths is much faster. The unsigned parental field is 
# increased by 1 so we can represent background as zero. So a value means
# voxel+1. Use path_from_parents to compute a path from the source to a target.

#parents = dijkstra3d.parental_field(field, source=(0,0,0), connectivity=6) # default is 26 connected

#path = dijkstra3d.path_from_parents(parents, target=(511, 511, 511))

#print(path.shape)

# Given a boolean label "field" and a source vertex, compute 
# the anisotropic euclidean distance from the source to all labeled vertices.

#dist_field = dijkstra3d.euclidean_distance_field(field, source=(0,0,0), anisotropy=(4,4,40))

# To make the EDF go faster add the free_space_radius parameter. It's only
# safe to use if you know that some distance around the source point
# is unobstructed space. For that region, we use an equation instead
# of dijkstra's algorithm. Hybrid algorithm! free_space_radius is a physical
# distance, meaning you must account for anisotropy in setting it.

#dist_field = dijkstra3d.euclidean_distance_field(field, source=(0,0,0), anisotropy=(4,4,40), free_space_radius=300) 

# Given a numerical field, for each directed edge from adjacent voxels A and B, 
# use B as the edge weight. In this fashion, compute the distance from a source 
# point for all finite voxels.

#dist_field = dijkstra3d.distance_field(field, source=(0,0,0))

# You can also provide a voxel connectivity graph to provide customized
# constraints on the permissible directions of travel. The graph is a
# uint32 image of equal size that contains a bitfield in each voxel 
# where each of the first 26-bits describes whether a direction is 
# passable. The description of this field can be seen here: 
# https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92
#
# The motivation for this feature is handling self-touching labels, but there
# are many possible ways of using this.

#graph = np.zeros(field.shape, dtype=np.uint32)
#graph += 0xffffffff # all directions are permissible
#graph[5,5,5] = graph[5,5,5] & 0xfffffffe # sets +x direction as impassable at this voxel
#path = dijkstra.dijkstra(..., voxel_graph=graph)
#print(path)
