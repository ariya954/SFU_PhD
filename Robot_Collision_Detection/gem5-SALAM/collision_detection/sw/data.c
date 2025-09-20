#include "data.h"

const float x_of_octant_relative_to_parent[8] = {-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5};

const float y_of_octant_relative_to_parent[8] = {-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5};

const float z_of_octant_relative_to_parent[8] = {-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5};

const int occupancy_status[73] = {                           PARTIALLY_OCCUPIED, 
EMPTY,       PARTIALLY_OCCUPIED,       EMPTY,       EMPTY,       PARTIALLY_OCCUPIED,       EMPTY,       PARTIALLY_OCCUPIED,       EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY,
EMPTY,       FULLY_OCCUPIED,           EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       FULLY_OCCUPIED,           EMPTY,       EMPTY,                    EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       FULLY_OCCUPIED,           EMPTY,
EMPTY,       EMPTY,                    EMPTY,       EMPTY,       EMPTY,                    EMPTY,       EMPTY,                    EMPTY};

const int robot_configuration[17] = {   2, /* The robot's width   */
                                        2, /* The robot's length  */
                                        2, /* The robot's height  */
                                        7, /* x of robot's center */
                                        7, /* y of robot's center */
                                        7, /* z of robot's center */
                                        0, /* The cosine of the angle between robot's width and x  */
                                        1, /* The cosine of the angle between robot's width and y  */
                                        0, /* The cosine of the angle between robot's width and z  */
                                        1, /* The cosine of the angle between robot's length and x */
                                        0, /* The cosine of the angle between robot's length and y */
                                        0, /* The cosine of the angle between robot's length and z */
                                        0, /* The cosine of the angle between robot's height and x */
                                        0, /* The cosine of the angle between robot's height and y */
                                        1, /* The cosine of the angle between robot's height and z */
                                        1, /* The robot's inscribed sphere radius */
                                        1  /* The robot's bounding sphere radius  */}; 
