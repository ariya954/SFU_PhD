//Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <stdint.h>
#include "../defines.h"
#include "../collision_detection_clstr_hw_defines.h"

#define DEV_INIT	0x01
#define DEV_INTR	0x04

#define NUMBER_OF_OCTREE_NODES 1460 // 73 nodes, each node has 5 4-bytes values (occupancy status, size and 3 for center) => 73*5*4 = 1460 
