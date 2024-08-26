from python_support.cmodel import *


class BASE_DATA(Structure):
    _fields_ = [("i8_data", POINTER(c_int8)),
                ("u8_data", POINTER(c_uint8)),
                ("i32_data", POINTER(c_int32))]


class MEMORY_ADDRES_MAP(Structure):
    _fields_ = [("buffer_addr", c_int),
                ("buffer_size", c_int),
                ("memory_type", c_int),
                ("data_type", c_int),
                ("data", BASE_DATA)]

    def get_data(self):

        data_arr = []
        if self.data_type == 0:
            data_ptr = self.data.i8_data

        if self.data_type == 1:
            data_ptr = self.data.u8_data

        if self.data_type == 2:
            data_ptr = self.data.i32_data

        for i in range(self.buffer_size):
            data_arr.append(data_ptr[i])

        return data_arr


class WEIGHT_READ_PARAM(Structure):
    _fields_ = [("addr", c_int),
                ("len", c_int64)]


class DMA_PARAM(Structure):
    _fields_ = [("psram_addr", c_int),
                ("psram_len", c_int),
                ("shared_memory_addr", c_int),
                ("shared_memory_len", c_int),
                ("dma_type", c_int)]


class SHARED_MEMORY(Structure):
    _fields_ = [("memory_size", c_int),
                ("bank_num", c_int),
                ("i64_data", POINTER(c_int64))]


class LINE_BUFFER(Structure):
    _fields_ = [("memory_size", c_int),
                ("i64_data", POINTER(c_int64))]


class PSRAM(Structure):
    _fields_ = [("memory_size", c_int),
                ("i64_data", POINTER(c_int64))]


class MEMORY_DATA(Structure):
    _fields_ = [("shared_memory", SHARED_MEMORY),
                ("line_buffer", LINE_BUFFER),
                ("psram", PSRAM)]


class MEMORY(Structure):
    _fields_ = [("memory_data", MEMORY_DATA),
                ("input", MEMORY_ADDRES_MAP),
                ("output", MEMORY_ADDRES_MAP)]


memory_init = lib.memory_init
memory_init.argtypes = [POINTER(MEMORY)]
memory_init.restype = None

memory_dma = lib.memory_dma
memory_dma.argtypes = [POINTER(MEMORY), DMA_PARAM]
memory_dma.restype = None

memory_read = lib.memory_read
memory_read.argtypes = [POINTER(MEMORY), MEMORY_ADDRES_MAP]
memory_read.restype = None

memory_write = lib.memory_write
memory_write.argtypes = [POINTER(MEMORY), MEMORY_ADDRES_MAP]
memory_write.restype = None

memory_destory = lib.memory_destory
memory_destory.argtypes = [POINTER(MEMORY)]
memory_destory.restype = None

if __name__ == "__main__":
    # py_values = [100]
    # base_data = BASE_DATA()
    # i32_input_data = (c_int32 * len(py_values))(*py_values)
    # base_data.i32_data = i32_input_data
    # input = MEMORY_ADDRES_MAP(0, 1, 0, 0, base_data)

    py_values = [100]
    input_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(input_data)
    input = MEMORY_ADDRES_MAP(0, 1, 0, 0, base_data)

    memory = MEMORY()
    memory_init(byref(memory))

    psram_memory_size = memory.memory_data.psram.memory_size
    py_values = [1] * psram_memory_size
    psram_data = (c_int64 * len(py_values))(*py_values)
    memmove(memory.memory_data.psram.i64_data, psram_data, sizeof(psram_data))

    ##dma psram to shared_memory
    dma_before_share_memory_data_list = [memory.memory_data.shared_memory.i64_data[i] for i in range(10)]
    dma_param = DMA_PARAM(0, 10, 0, 10, 0)
    memory_dma(byref(memory), dma_param)
    dma_after_share_memory_data_list = [memory.memory_data.shared_memory.i64_data[i] for i in range(10)]

    ##dma shared_memory to psram
    dma_before_psram_data_list = [memory.memory_data.psram.i64_data[i] for i in range(10)]
    dma_param = DMA_PARAM(0, 10, 10, 10, 1)
    memory_dma(byref(memory), dma_param)
    dma_after_psram_data_list = [memory.memory_data.psram.i64_data[i] for i in range(10)]

    ##write data from memory
    py_values = [1] * 8
    input_data = (c_int8 * len(py_values))(*py_values)
    base_data = BASE_DATA(input_data)
    buffer_len = len(py_values)
    input = MEMORY_ADDRES_MAP(10, buffer_len, 0, 0, base_data)
    memory_write(byref(memory), input)

    ##read data from memory
    output_data = (c_int8 * buffer_len)()
    base_data = BASE_DATA(output_data)
    output = MEMORY_ADDRES_MAP(10, buffer_len, 0, 0, base_data)
    memory_read(byref(memory), output)

    memory_destory(byref(memory))
    print("---------------------------------------")
