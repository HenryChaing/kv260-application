#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "npu_project.h"


int main()
{

    int ret = 0;
    int fd = open(RTOS_CMDQU_DEV_PATH, O_RDWR);
    if(fd <= 0)
    {
        printf("open failed! fd = %d\n", fd);
        return 0;
    }

    unsigned short hw_input[16] = {0xfdf4, 0x01e2, 0x018a, 0x0047, 0xffd5, 0xff6e, 0xfc86, 0xfdc9, 0xfc86, 0xfcdd, 0xfcc3, 0xfc9b, 0xfc23,0xfca8, 0x00bf, 0x01dc};
    struct ioctl_arg array = {0};
    for (int i = 0; i < 16; i++) {
		array.npu_array[i] = hw_input[i];
	} 

    struct cmdqu_t cmd = {0};
    // cmd.ip_id = 0;
    // cmd.cmd_id = CMD_DUO_LED;
    // cmd.resv.mstime = 100;
    // cmd.param_ptr = DUO_LED_ON;
    cmd.input_stream_address = 0x7fe00000;
    cmd.output_stream_address = 0x7fe00100;  

    ret = ioctl(fd , RTOS_NPU_SET_DATA_VALUE, &cmd);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }
    sleep(1);
    printf("A53: write stream address\n");

    sleep(1);

    cmd.input_stream_address =  0;
    cmd.output_stream_address = 0;

    ret = ioctl(fd , RTOS_NPU_GET_DATA_VALUE, &cmd);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }

    sleep(1);
    printf("A53: get stream address\n\n");
    printf("Input Stream Addr: %lu \n Output Stream Addr: %lu \n",cmd.input_stream_address,cmd.output_stream_address);

    ret = ioctl(fd , RTOS_NPU_SET_SHMEM_VALUE, &array);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }

    cmd.cmd_id = 1;
    ret = ioctl(fd , RTOS_NPU_SET_CTRL_VALUE, &cmd);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }
    sleep(1);
    printf("A53: write control value\n");

    cmd.cmd_id = 1;
    ret = ioctl(fd , RTOS_NPU_GET_CTRL_VALUE, &cmd);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }
    sleep(1);
    printf("A53: write control value %d\n",cmd.cmd_id);
    
    ret = ioctl(fd , RTOS_NPU_GET_SHMEM_VALUE, &array);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }
    
    for (int i = 0; i < 16; i++) {
		printf("array[%d]: %d, ",i ,array.npu_array[i]);
	}   printf("\n");

    close(fd);
    return 0;
}
