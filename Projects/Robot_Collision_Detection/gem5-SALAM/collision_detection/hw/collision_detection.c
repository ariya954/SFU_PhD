#include "hw_defines.h"

void check_intersection_between_robot_and(int index_of_octree_of_environment) {

    uint8_t * octreeBase = (uint8_t *)OCTREE_OF_ENVIRONMENT;
    uint8_t * robotObjectBoundingBoxBase = (uint8_t *)(OBJECT_BOUNDING_BOX_OF_ROBOT);
    uint8_t * result_of_collision = (uint8_t *)(RESULT_OF_COLLISION);    
    TYPE    * octree_of_environment = (TYPE    *)octreeBase;
    TYPE   * robot_object_bounding_box = (TYPE    *)robotObjectBoundingBoxBase;
    TYPE    * collision_result = (TYPE    *)result_of_collision;

    // The minimum and maximum x, y and z of the space, which are occupied by the environment, which is a cube, is: center +- size/2
    int x_min_environment = octree_of_environment[index_of_octree_of_environment+2] - (octree_of_environment[index_of_octree_of_environment+1] / 2);
    int x_max_environment = octree_of_environment[index_of_octree_of_environment+2] + (octree_of_environment[index_of_octree_of_environment+1] / 2);
    int y_min_environment = octree_of_environment[index_of_octree_of_environment+3] - (octree_of_environment[index_of_octree_of_environment+1] / 2);
    int y_max_environment = octree_of_environment[index_of_octree_of_environment+3] + (octree_of_environment[index_of_octree_of_environment+1] / 2);
    int z_min_environment = octree_of_environment[index_of_octree_of_environment+4] - (octree_of_environment[index_of_octree_of_environment+1] / 2);
    int z_max_environment = octree_of_environment[index_of_octree_of_environment+4] + (octree_of_environment[index_of_octree_of_environment+1] / 2);
    
    // The minimum and maximum x, y, z of the space, which are inside the bounding sphere of robot, is: center +- bounding sphere radius
    int x_min_bounding_sphere = robot_object_bounding_box[3] - robot_object_bounding_box[16];
    int x_max_bounding_sphere = robot_object_bounding_box[3] + robot_object_bounding_box[16];
    int y_min_bounding_sphere = robot_object_bounding_box[4] - robot_object_bounding_box[16];
    int y_max_bounding_sphere = robot_object_bounding_box[4] + robot_object_bounding_box[16];
    int z_min_bounding_sphere = robot_object_bounding_box[5] - robot_object_bounding_box[16];
    int z_max_bounding_sphere = robot_object_bounding_box[5] + robot_object_bounding_box[16];
    
    // If there is no overlap between the cube of the environment and the bounding sphere of robot, then they have no intersection
    if((x_min_environment >= x_max_bounding_sphere) || (x_max_environment <= x_min_bounding_sphere))
       if((y_min_environment >= y_max_bounding_sphere) || (y_max_environment <= y_min_bounding_sphere))
          if((z_min_environment >= z_max_bounding_sphere) || (z_max_environment <= z_min_bounding_sphere)) {
             *collision_result = 0;
             return;
          }
          
    // The minimum and maximum x, y, z of the space, which are inside the inscribed sphere of robot, is: center +- inscribed sphere radius
    int x_min_inscribed_sphere = robot_object_bounding_box[3] - robot_object_bounding_box[15];
    int x_max_inscribed_sphere = robot_object_bounding_box[3] + robot_object_bounding_box[15];
    int y_min_inscribed_sphere = robot_object_bounding_box[4] - robot_object_bounding_box[15];
    int y_max_inscribed_sphere = robot_object_bounding_box[4] + robot_object_bounding_box[15];
    int z_min_inscribed_sphere = robot_object_bounding_box[5] - robot_object_bounding_box[15];
    int z_max_inscribed_sphere = robot_object_bounding_box[5] + robot_object_bounding_box[15];
   
    // If the cube of the environment and the inscribed sphere of robot overlap in all dimensions, then they have intersection
    if(((x_min_environment <= x_min_inscribed_sphere) && (x_min_inscribed_sphere <= x_max_environment)) || ((x_min_environment <= x_max_inscribed_sphere) && (x_max_inscribed_sphere <= x_max_environment)))
       if(((y_min_environment <= y_min_inscribed_sphere) && (y_min_inscribed_sphere <= y_max_environment)) || ((y_min_environment <= y_max_inscribed_sphere) && (y_max_inscribed_sphere <= y_max_environment)))
          if(((z_min_environment <= z_min_inscribed_sphere) && (z_min_inscribed_sphere <= z_max_environment)) || ((z_min_environment <= z_max_inscribed_sphere) && (z_max_inscribed_sphere <= z_max_environment))) {
             *collision_result = 1; 
             return;
          }   
          
    // The minimum and maximum x, y, z of the space, which are occupied by the robot, which is a box, is:
    // 							center +- length image robot on axes(summation(orientation * size))/2
   
    int length_image_robot_on_x = (robot_object_bounding_box[6]*robot_object_bounding_box[0]) +(robot_object_bounding_box[9]*robot_object_bounding_box[1]) + (robot_object_bounding_box[12]*robot_object_bounding_box[2]); 
    int length_image_robot_on_y = (robot_object_bounding_box[7]*robot_object_bounding_box[0]) +(robot_object_bounding_box[10]*robot_object_bounding_box[1]) + (robot_object_bounding_box[13]*robot_object_bounding_box[2]); 
    int length_image_robot_on_z = (robot_object_bounding_box[8]*robot_object_bounding_box[0]) +(robot_object_bounding_box[11]*robot_object_bounding_box[1]) + (robot_object_bounding_box[14]*robot_object_bounding_box[2]); 
                                                                                                       
    int x_min_robot = robot_object_bounding_box[3] - (length_image_robot_on_x / 2);
    int x_max_robot = robot_object_bounding_box[3] + (length_image_robot_on_x / 2);
    int y_min_robot = robot_object_bounding_box[4] - (length_image_robot_on_y / 2);
    int y_max_robot = robot_object_bounding_box[4] + (length_image_robot_on_y / 2);
    int z_min_robot = robot_object_bounding_box[5] - (length_image_robot_on_z / 2);
    int z_max_robot = robot_object_bounding_box[5] + (length_image_robot_on_z / 2);
    
    // If the cube of the environment and the robot overlap in all dimensions, then they have intersection
    if(((x_min_environment <= x_min_robot) && (x_min_robot <= x_max_environment)) || ((x_min_environment <= x_max_robot) && (x_max_robot <= x_max_environment)))
       if(((y_min_environment <= y_min_robot) && (y_min_robot <= y_max_environment)) || ((y_min_environment <= y_max_robot) && (y_max_robot <= y_max_environment)))
          if(((z_min_environment <= z_min_robot) && (z_min_robot <= z_max_environment)) || ((z_min_environment <= z_max_robot) && (z_max_robot <= z_max_environment)))
             *collision_result = 1;
}

void top() {

    uint8_t * octreeBase = (uint8_t *)OCTREE_OF_ENVIRONMENT;
    uint8_t * robotObjectBoundingBoxBase = (uint8_t *)(OBJECT_BOUNDING_BOX_OF_ROBOT);
    uint8_t * result_of_collision = (uint8_t *)(RESULT_OF_COLLISION);    
    TYPE    * octree_of_environment = (TYPE    *)octreeBase;
    TYPE   * robot_object_bounding_box = (TYPE    *)robotObjectBoundingBoxBase;
    TYPE    * collision_result = (TYPE    *)result_of_collision;   
    
    // To detect collision, first, root of the environment octree is checked by setting current_node to 0
    int current_node = 0;
    // For each child node (each node of octree has 8 children nodes)
    for(int i = 0; i < 8; i++) {
        // If the child node ((current_node*8)+i+1) is fully occupied and there is an intersection between the child node of
        // the environment and robot, then there is a collision between robot and environment at the child's position
        if(octree_of_environment[((current_node*8)+i+1) * 5] == FULLY_OCCUPIED) {
           check_intersection_between_robot_and(((current_node*8)+i+1)*5);
           if(*collision_result == 1) {
              *collision_result = 1;
              return;
           }
        }
        // If the child node ((current_node*8)+i+1) is partially occupied and there is an intersection between the child node
        // of the environment and robot, then the child node's children should be checked to check if one of its children is fully
        // occupied and have collision with robot, thus current_node is set to the child node ((current_node*8) + i + 1) and i to -1
        // so that by increasing i by the for loop, i gets 0 and thus, checking the children nodes is done from the first child (i = 0) 
        if(octree_of_environment[((current_node*8)+i+1) * 5] == PARTIALLY_OCCUPIED) {
           check_intersection_between_robot_and(((current_node*8)+i+1)*5);
           if(*collision_result == 1) {
              current_node = (current_node*8) + i + 1;
              i = -1;
           }
        }
        // If i gets 7, it means that no collision is found for any child nodes of the current_node  
        if(i == 7) {
           // If current_node is root, so checking the environment octree is done and robot and environment have no collision  
           if(current_node == 0) {
              *collision_result = 0;
              return;
           }
           // Otherwise, back to the parent node ((current_node - 1) / 8) of current_node and check the remaining child nodes of the
           // parent node by setting i to current_node % 8 (example: if the current_node is 17, thus its parent is node 2 and if no
           // collision is found for node 17, we should back to node 2 (parent) and continue checking for nodes 18 to 24 (other childs of
           // parent node) so by setting i to 17%8 = 1, checking is first done for node (current_node*8)+i+1) = ((2*8)+1+1) = 18, which
           // is the adjacent child node of node 17)
           else {
              i = (current_node - 1) % 8;
              current_node = (current_node - 1) / 8;
           } 
        }
    }  
}
