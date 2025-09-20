#include "hw_defines.h"

void top() {

	//Define Device MMRs
	volatile uint8_t  * COLLISION_DETECTION_Flags = (uint8_t *)COLLISION_DETECTION;
	volatile uint8_t  * DmaFlags   = (uint8_t  *)(DMA_Flags);
	volatile uint64_t * DmaRdAddr  = (uint64_t *)(DMA_RdAddr);
	volatile uint64_t * DmaWrAddr  = (uint64_t *)(DMA_WrAddr);
	volatile uint32_t * DmaCopyLen = (uint32_t *)(DMA_CopyLen);

	//Transfer Result
	*DmaRdAddr  = ENVIRONMENT_OCTREE;
	*DmaWrAddr  = OCTREE_OF_ENVIRONMENT;
	*DmaCopyLen = NUMBER_OF_OCTREE_NODES;
	*DmaFlags   = DEV_INIT;
	while ((*DmaFlags & DEV_INTR) != DEV_INTR);
	
	//Transfer Result
	*DmaRdAddr  = ROBOT_OBJECT_BOUNDING_BOX;
	*DmaWrAddr  = OBJECT_BOUNDING_BOX_OF_ROBOT;
	*DmaCopyLen = 68; // 17 values of 32-bit or 4 bytes integer
	*DmaFlags   = DEV_INIT;
	while ((*DmaFlags & DEV_INTR) != DEV_INTR);

	//Start the accelerated function
	*COLLISION_DETECTION_Flags = DEV_INIT;
	//Poll function for finish
	while ((*COLLISION_DETECTION_Flags & DEV_INTR) != DEV_INTR);

	//Transfer Result
	*DmaRdAddr  = RESULT_OF_COLLISION;
	*DmaWrAddr  = COLLISION_RESULT;
	*DmaCopyLen = 4;
	*DmaFlags   = DEV_INIT;
	//Poll DMA for finish
	while ((*DmaFlags & DEV_INTR) != DEV_INTR);
	return;
}
