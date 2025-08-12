#define RTOS_CMDQU_DEV_PATH "/dev/xlnx-design-newip"

union ap_ctrl_regs_offset{
	unsigned int ap_ctrl_regs;
	int reserved;
};

union gie_offset{
	unsigned char gie;
	int reserved;
};

union ipie_offset{
	unsigned char ipie;
	int reserved;
};

union ipis_offset{
	unsigned char ipis;
	int reserved;
};

/* register mapping refers to mailbox user guide*/
struct newip_control_register{
	union ap_ctrl_regs_offset   ap_ctrl_regs_offset;    //0x00, 0x04, 0x08, 0x0c
	union gie_offset            gie_offset;             //0x10~0x1C, 0x20~0x2C, 0x30~0x3C, 0x40~0x4C
	union ipie_offset           ipie_offset;            //0x50~0x5C
	union ipis_offset           ipis_offset;            //0x60
};

struct stream_address {
	unsigned int lower;
	unsigned int upper;
};

struct newip_data_register{
	struct stream_address            input_stream;    //0x00, 0x04, 0x08, 0x0c
	unsigned int					 reserved; 
	struct stream_address            output_stream;             //0x10~0x1C, 0x20~0x2C, 0x30~0x3C, 0x40~0x4C
};

struct newip_register{
	struct newip_control_register   newip_control_register;    //0x00, 0x04, 0x08, 0x0c
	struct newip_data_register      newip_data_register;       //0x10~0x1C, 0x20~0x2C, 0x30~0x3C, 0x40~0x4C
};

volatile struct newip_register *newip_regs;

#define MAILBOX_MAX_NUM         0x0008
#define MAILBOX_DONE_OFFSET     0x0002
#define MAILBOX_CONTEXT_OFFSET  0x0400

/************ rtos_cmdqu.h part ************/

typedef struct cmdqu_t cmdqu_t;
/* cmdqu size should be 8 bytes because of mailbox buffer size */
struct cmdqu_t {
	unsigned char cmd_id ;
	unsigned long input_stream_address;
	unsigned long output_stream_address;
} __attribute__((packed)) __attribute__((aligned(0x8)));

struct ioctl_arg { 
    unsigned short npu_array[16]; 
};

/* keep those commands for ioctl system used */
enum SYSTEM_CMD_TYPE {
	CMDQU_SEND = 1,
	CMDQU_SEND_WAIT,
	CMDQU_SEND_WAKEUP,
	NPU_SET_DATA_VALUE,
    NPU_GET_DATA_VALUE,
	NPU_SET_CTRL_VALUE,
    NPU_GET_CTRL_VALUE,
	NPU_SET_SHMEM_VALUE,
	NPU_GET_SHMEM_VALUE,
};

#define RTOS_CMDQU_SEND                         _IOW('r', CMDQU_SEND, unsigned long)
#define RTOS_CMDQU_SEND_WAIT                    _IOW('r', CMDQU_SEND_WAIT, unsigned long)
#define RTOS_CMDQU_SEND_WAKEUP                  _IOW('r', CMDQU_SEND_WAKEUP, unsigned long)
#define RTOS_NPU_SET_DATA_VALUE                      _IOW('r', NPU_SET_DATA_VALUE, unsigned long)
#define RTOS_NPU_GET_DATA_VALUE                      _IOW('r', NPU_GET_DATA_VALUE, unsigned long)
#define RTOS_NPU_SET_CTRL_VALUE                      _IOW('r', NPU_SET_CTRL_VALUE, unsigned long)
#define RTOS_NPU_GET_CTRL_VALUE                      _IOW('r', NPU_GET_CTRL_VALUE, unsigned long)
#define RTOS_NPU_SET_SHMEM_VALUE                     _IOW('r', NPU_SET_SHMEM_VALUE, unsigned long)
#define RTOS_NPU_GET_SHMEM_VALUE                     _IOW('r', NPU_GET_SHMEM_VALUE, unsigned long)
