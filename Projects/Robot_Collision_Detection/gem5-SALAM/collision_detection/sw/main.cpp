#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bench.h"
#include "data.c"
#include "../../../common/m5ops.h"
#include "../collision_detection_clstr_hw_defines.h"

volatile uint8_t  * top   = (uint8_t  *) (TOP + 0x00);
volatile uint32_t * environment_octree = (uint32_t *) (ENVIRONMENT_OCTREE);
volatile uint32_t * robot_object_bounding_box = (uint32_t *) (ROBOT_OBJECT_BOUNDING_BOX);
volatile uint32_t * collision_result = (uint32_t *) (COLLISION_RESULT);

int pow(int a, int b) {
    int result = 1;
    for(int i = 0; i < b; i++)
        result = result * a;
    return result;
}

int __attribute__ ((optimize("0"))) main(void) {
	m5_reset_stats();

    // Initializing the accelerator's input
    
    // Starting the initialization from the root of the environment octree (environment_octree[0] : roots' occupancy status
    //                                                                      environment_octree[1] : root's size
    //                                                                      environment_octree[2:4] : root's center)
    environment_octree[0] = occupancy_status[0]; 
    environment_octree[1] = OCTREE_ROOT_SIZE;     // Root represents the entire environment, thus its size is the environment size, which
                                                  // is a cube
    environment_octree[2] = OCTREE_ROOT_SIZE / 2; // x of the center of the envirnoment, which is a cube, is the half of the cube size
    environment_octree[3] = OCTREE_ROOT_SIZE / 2; // y of the center of the envirnoment, which is a cube, is the half of the cube size
    environment_octree[4] = OCTREE_ROOT_SIZE / 2; // z of the center of the envirnoment, which is a cube, is the half of the cube size
 
    // Traversing the environment octree and initializing the environment   
    int current_node = 1;
    int current_node_size = OCTREE_ROOT_SIZE / 2;
    for(int i = 1; i < OCTREE_DEPTH; i++) { 
    	for(int j = current_node; current_node < j + pow(8, i); current_node++) {
    	    environment_octree[(current_node*5) + 0] = occupancy_status[current_node]; 
    	    environment_octree[(current_node*5) + 1] = current_node_size;
    	    // Center of node = center of node's parent (node - 1 /8) + node's relative position to parent * node size 
    	    environment_octree[(current_node*5) + 2] = environment_octree[(((current_node-1)/8) * 5) + 2] + (x_of_octant_relative_to_parent[(current_node-1)%8] * current_node_size); 
    	    environment_octree[(current_node*5) + 3] = environment_octree[(((current_node-1)/8) * 5) + 3] + (y_of_octant_relative_to_parent[(current_node-1)%8] * current_node_size);
    	    environment_octree[(current_node*5) + 4] = environment_octree[(((current_node-1)/8) * 5) + 4] + (z_of_octant_relative_to_parent[(current_node-1)%8] * current_node_size);  	        	    	
    	}
    	current_node_size = current_node_size / 2; // The size of nodes are half of their parents, because we devide the area into 8 parts
    }
    
    // Initializing the robot's size, center, orientation, inscribed and bounding sphere radiuses by robot_configuaration in data.c 
    for(int i = 0; i < 17; i++)
    	robot_object_bounding_box[i] = robot_configuration[i];

    // Run the top level function
    volatile int count = 0;
    stage = 0;
    *top = 0x01;
    while (stage < 1) count++;
    
    // Check if the accelerator's output is correct
    #ifdef CHECK
    bool fail = false;
    if(*collision_result != 1){
    	printf("ERROR: The accelerator did not detect the collision\n");
        fail = true;
    }
    if(fail)
        printf("Check Failed\n");
    else
        printf("Check Passed\n");
    #endif
    printf("Job complete\n");
	m5_dump_stats();
	m5_exit();
}
