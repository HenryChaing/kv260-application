#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "npu_project.h"
#include "npy_array.h"

int main(int argc, char *argv[])
{
/*** INIT part ***/
    int ret = 0;
    int fd = open(RTOS_CMDQU_DEV_PATH, O_RDWR);
    if(fd <= 0)
    {
        printf("open failed! fd = %d\n", fd);
        return 0;
    }

    struct cmdqu_t cmd = {0};
    cmd.input_stream_address = 0x7fe00000;
    cmd.output_stream_address = 0x7fe00100;  

    ret = ioctl(fd , RTOS_NPU_SET_DATA_VALUE, &cmd);
    if(ret < 0)
    {
        printf("ioctl error!\n");
        close(fd);
    }
    printf("A53: write stream address\n");
    sleep(1);
    

/*** NPY part ***/

    if( argc != 2 ){
        printf("Usage: %s <--train/--test>\n", argv[0]);
        return EXIT_FAILURE;
    }

    npy_array_t *arr_x, *arr_y;
    short *array_x, *array_y;
    int n_rows, n_cols;
    int max_id_output = -1, max_id_y = -1;
    int max_val_output = -1, max_val_y = -1;
    int score = 0;

    if (!strcmp(argv[1],"--train")){
        arr_x = npy_array_load("X_train_int8_val.npy");
        arr_y = npy_array_load("y_train_int8_val.npy");
    } else if (!strcmp(argv[1],"--test")){
        arr_x = npy_array_load("X_test_int8_val.npy");
        arr_y = npy_array_load("y_test_int8_val.npy");        
    }

    if( !arr_x || !arr_y){
        printf("Cannot read NumPy file '%s'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    npy_array_dump( arr_x );
    npy_array_dump( arr_y );

    if( (arr_x->typechar == 'i' && arr_x->elem_size == 2) ){
        array_x = (short*) arr_x->data;
    } else {
        printf("Try with a npy array of 2 dimensions and 'float32' precision elements.\n");
        return EXIT_FAILURE;
    }

    if( (arr_y->typechar == 'i' && arr_y->elem_size == 2) ){
        array_y = (short*) arr_y->data;
    } else {
        printf("Try with a npy array of 2 dimensions and 'float32' precision elements.\n");
        return EXIT_FAILURE;
    }

    n_rows = arr_x->shape[0];
    // n_rows = 3;
    n_cols = arr_x->shape[1];
        
/*** NPU part ***/
    struct ioctl_arg arg_content = {0};
    for (int turn=0; turn < n_rows; turn++) {
        
        for (int i = 0; i < n_cols; i++) {
            arg_content.npu_array[i] = *(array_x + turn * n_cols + i);
        }
        
        ret = ioctl(fd , RTOS_NPU_SET_SHMEM_VALUE, &arg_content);
        if(ret < 0)
        {
            printf("RTOS_NPU_SET_SHMEM_VALUE error!\n");
            close(fd);
            return 1;
        }

        cmd.cmd_id = 1;
        ret = ioctl(fd , RTOS_NPU_SET_CTRL_VALUE, &cmd);
        if(ret < 0)
        {
            printf("RTOS_NPU_SET_CTRL_VALUE error!\n");
            close(fd);
            return 1;
        }
        // printf("A53: write control value\n");

        cmd.cmd_id = 1;
        ret = ioctl(fd , RTOS_NPU_GET_CTRL_VALUE, &cmd);
        if(ret < 0)
        {
            printf("RTOS_NPU_GET_CTRL_VALUE error!\n");
            close(fd);
            return 1;
        }
        // printf("A53: write control value %d\n",cmd.cmd_id);

        ret = ioctl(fd , RTOS_NPU_GET_SHMEM_VALUE, &arg_content);
        if(ret < 0)
        {
            printf("RTOS_NPU_GET_SHMEM_VALUE error!\n");
            close(fd);
            return 1;
        }
        
        for (int i = 0; i < 5; i++) {    
            if (arg_content.npu_array[i] > max_val_output){
                max_id_output = i;
                max_val_output = arg_content.npu_array[i];
            }
            // printf("%x\n", *(array_y + turn * n_cols + i));
            if (*(array_y + turn * n_cols + i) > max_val_y){
                max_id_y = i;
                max_val_y = *(array_y + turn * n_cols + i);
            }
        }
        max_val_output = 0;max_id_output = 0;
        max_val_y = 0; max_id_y = 0;
        // printf("----\n");
        if (max_id_output == max_id_y){
            score++;
        }
    }
    printf("accuracy: %d\n", score*100/n_rows);

    npy_array_free( arr_x );
    npy_array_free( arr_y );
    close(fd);
    return 0;
}
