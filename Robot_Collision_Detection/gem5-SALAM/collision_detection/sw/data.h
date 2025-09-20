#include "../defines.h"

#define OCTREE_DEPTH 3
#define OCTREE_ROOT_SIZE 8

extern const float x_of_octant_relative_to_parent[8];

extern const float y_of_octant_relative_to_parent[8];

extern const float z_of_octant_relative_to_parent[8];

extern const int occupancy_status[73]; // Number of nodes of octree: summation 8^i, i = 0 : Depth - 1 => 73 = (8^0) + (8^1) + (8^2)

extern const int robot_configuration[17]; // 3 for size, 3 for center, 3*3 for orientation, 2 for inscribed and bounding sphere radiuses
