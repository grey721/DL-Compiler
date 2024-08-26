from ir.dialect.npu.IR_operator import *
from ir.dialect.npu.IR_memory import *
import math


class ada200:
    shared_memory = SharedMemory()
    # psram = Psram(8*1024)
    cim_row = 256
    cim_subcell = 4
    cim_col = 16
    cluster_cim_num = 4
    cluster_num = 4
    fmi_max_size = 512 * 1024
    # first_layer_fmi_max_size = 256*1024
    first_layer_fmi_max_size = 128 * 1024
    fmo_max_size = 512 * 1024
    short_cut_out_max_size = 512 * 1024
    concat_out_max_size = 512 * 1024
    cluster_max_hwc = 16 * 256
    vpu_max_buffer_size = 512 * 128 / 8
    vpu_tile_channel_list = [16, 32, 64]
    vpu_block_channel_list = [16, 32, 64, 128, 256]
    bank_group_id_list = [0, 1, 2]
    max_line_buffer = 16 * 4  # kb

    def __init__(self):
        pass

    def get_other_param(self, conv_type, kernel_h, kernel_w,
                        kernel_cmod, kernel_n):

        kernel_hwcmod = kernel_h * kernel_w * kernel_cmod

        # calculate the cim_psum
        if conv_type == 1:
            cim_psum = 1
        elif kernel_hwcmod <= int(64 * 4 / 8):
            cim_psum = 1
        elif (kernel_hwcmod > int(64 * 4 / 8)) & (kernel_hwcmod <= int(64 * 8 / 8)):
            cim_psum = 2
        elif kernel_hwcmod > int(64 * 8 / 8):
            cim_psum = 4

        if cim_psum == 1:
            if math.ceil(kernel_n / 16) == 1:
                kernel_m = 1
            elif math.ceil(kernel_n / 16) == 2:
                kernel_m = 2
            else:
                kernel_m = 4
        elif cim_psum == 2:
            if math.ceil(kernel_n / 16) == 1:
                kernel_m = 1
            else:
                kernel_m = 2
        else:
            kernel_m = 1

        # calculate the cluster_win_num
        if conv_type == 1:
            cluster_win_num = 1

        elif kernel_hwcmod <= int(64 * 4 / 8):
            if kernel_m == 1:
                cluster_win_num = 4
            elif kernel_m == 2:
                cluster_win_num = 2
            elif (kernel_m == 3) | (kernel_m == 4):
                cluster_win_num = 1

        elif (kernel_hwcmod > int(64 * 4 / 8)) & (kernel_hwcmod <= int(64 * 8 / 8)):
            if kernel_m == 1:
                cluster_win_num = 2
            elif kernel_m == 2:
                cluster_win_num = 1

        elif kernel_hwcmod > int(64 * 8 / 8):
            cluster_win_num = 1

        # calculate the num_slice
        if conv_type == 1:
            num_slice = 0
        elif kernel_hwcmod <= int(64 * 4 / 8):
            num_slice = kernel_hwcmod
        elif (kernel_hwcmod > int(64 * 4 / 8)) & (kernel_hwcmod <= int(64 * 8 / 8)):
            if (kernel_hwcmod & 1) == 0:
                num_slice = kernel_hwcmod / 2
            else:
                num_slice = kernel_hwcmod / 2 + 1
        elif kernel_hwcmod > int(64 * 8 / 8):
            if kernel_hwcmod % 4 == 0:
                num_slice = kernel_hwcmod / 4
            else:
                num_slice = kernel_hwcmod / 4 + 1
        num_slice = int(num_slice)

        return cluster_win_num, cim_psum, num_slice

    def get_weight_mapping_param(self, conv_type, kernel_n, kernel_h, kernel_w,
                                 kernel_c_range, output_tensor_info, first_layer=False):

        if not first_layer:
            if conv_type == 0:
                kernel_hwc = kernel_h * kernel_w * (kernel_c_range[1] - kernel_c_range[0])
                sub_cell_num = 4
                cim_base_row = 64
                cim_rows = sub_cell_num * cim_base_row
                hwc_cim_num = math.ceil(kernel_hwc / cim_rows)
                hwc_cluster_num = math.ceil(hwc_cim_num / 16)
                if hwc_cluster_num > 1:
                    cluster_psum = 4 * hwc_cluster_num
                    cluster_kernel_n = [4, 16]

                if hwc_cluster_num == 1:
                    if hwc_cim_num <= 4:
                        cluster_psum = 1
                    elif hwc_cim_num <= 8:
                        cluster_psum = 2
                    elif hwc_cim_num <= 16:
                        cluster_psum = 4

                # cal the pair between cluster_num and kernel_n
                if cluster_psum == 1:
                    if hwc_cim_num == 1:
                        cluster_kernel_n = [1, 64]
                    if hwc_cim_num == 2:
                        cluster_kernel_n = [1, 32]
                    if hwc_cim_num == 3 or hwc_cim_num == 4:
                        cluster_kernel_n = [1, 16]

                if cluster_psum == 2:
                    cluster_kernel_n = [2, 16]

                if cluster_psum == 4:
                    cluster_kernel_n = [4, 16]

                kernel_cmod = int((kernel_c_range[1] - kernel_c_range[0]) / 8)
                cluster_win_num, cim_psum, num_slice = \
                    self.get_other_param(conv_type, kernel_h, kernel_w, kernel_cmod, kernel_n)

                if cluster_psum <= 1:
                    cl_num_by_n = math.ceil(kernel_n / cluster_kernel_n[1] * cluster_kernel_n[0])
                    if cl_num_by_n == 1:
                        tile_num = 4
                    elif cl_num_by_n == 2:
                        tile_num = 2
                    elif cl_num_by_n >= 3:
                        tile_num = 1

                elif cluster_psum <= 2:
                    tile_num = 2

                elif cluster_psum <= 4:
                    tile_num = 4

                else:
                    tile_num = 1

                if cluster_psum == 1:
                    print(output_tensor_info['block_address_list'][1] * output_tensor_info['block_address_list'][3])
                    assert output_tensor_info['block_address_list'][1] * output_tensor_info['block_address_list'][
                        3] >= tile_num
                    if output_tensor_info['block_address_list'][1] > tile_num:
                        tile_split_mode_ins = tile_split_mode(tile_num, 1, 1)
                    else:
                        tile_split_mode_ins = tile_split_mode(int(tile_num / 2), int(tile_num / 2), 1)
                else:
                    tile_split_mode_ins = tile_split_mode(1, 1, cluster_psum)

                if hwc_cluster_num > 1:
                    bit_map = math.ceil(kernel_hwc / (hwc_cluster_num * 16) / 64)
                else:
                    if cluster_psum in [2, 4]:
                        bit_map = math.ceil(kernel_hwc / (cluster_psum * 4) / 64)
                    if cluster_psum == 1:
                        bit_map = math.ceil(kernel_hwc / (cim_psum) / 64)

                if bit_map == 3:
                    bit_map = 3

            if conv_type == 1:
                raise ValueError("not support dw_conv now")

        else:

            kernel_m = int(kernel_n / 16)
            if kernel_m == 1:
                cluster_win_num = 4
            elif kernel_m == 2:
                cluster_win_num = 2
            elif (kernel_m == 3) | (kernel_m == 4):
                cluster_win_num = 1

            if kernel_h == kernel_w == (kernel_c_range[1] - kernel_c_range[0]) == 3:

                cluster_psum = 1
                cim_psum = 1
                bit_map = 1
                tile_num = 1
                tile_split_mode_ins = tile_split_mode(1, 1, cluster_psum)
                num_slice = 0

            elif (kernel_h == kernel_w == 7) and ((kernel_c_range[1] - kernel_c_range[0]) == 3):

                cluster_psum = 1
                cim_psum = 1
                bit_map = 3
                tile_num = 1
                tile_split_mode_ins = tile_split_mode(1, 1, cluster_psum)
                num_slice = 0

            elif (kernel_h == kernel_w == 5) and ((kernel_c_range[1] - kernel_c_range[0]) == 3):

                cluster_psum = 1
                cim_psum = 1
                bit_map = 1
                tile_num = 1
                tile_split_mode_ins = tile_split_mode(1, 1, cluster_psum)
                num_slice = 0

            elif kernel_h == kernel_w == 1 and (kernel_c_range[1] - kernel_c_range[0]) == 3:

                cluster_psum = 1
                cim_psum = 1
                bit_map = 1
                tile_num = 1
                tile_split_mode_ins = tile_split_mode(1, 1, cluster_psum)
                num_slice = 0

        weight_mapping_dict = dict(cluster_psum=cluster_psum,
                                   cluster_win_num=cluster_win_num,
                                   cim_psum=cim_psum,
                                   bit_map=bit_map,
                                   num_slice=num_slice,
                                   tile_num=tile_num,
                                   tile_split_mode=tile_split_mode_ins)

        return weight_mapping_dict

    def get_tile_split_mode(self, block):
        npu_op_flow = block.NpuOp.NpuOpFlow
        npu_op_flow_block_address_list = block.npu_op_flow_block_address_list
        input_block_address_list = npu_op_flow_block_address_list[-1]["input_block_address_list"]
        ih = input_block_address_list[1]
        iw = input_block_address_list[3]
        ic = input_block_address_list[5]
        last_op = npu_op_flow[-1]

        assert len(npu_op_flow) <= 2
        th, tw, tc = 1, 1, 1

        if isinstance(last_op, NpuPool):

            while (iw / tw) * (ic / tc) > 64 * 256:
                tc += 1
                if tc > 4:
                    raise NotImplementedError

            assert ic % tc == 0

        block.tile_split_mode = tile_split_mode(th, tw, tc)

    def get_single_op_mapping_param(self, ih, iw, ic):

        cluster_psum = None
        cluster_win_num = None
        cim_psum = None
        bit_map = None
        num_slice = None
        assert ic == 64
        assert ih <= 128
        assert iw <= 128
        tile_num = 4
        tile_split_mode_ins = tile_split_mode(1, 1, 4)
        weight_mapping_dict = dict(cluster_psum=cluster_psum,
                                   cluster_win_num=cluster_win_num,
                                   cim_psum=cim_psum,
                                   bit_map=bit_map,
                                   num_slice=num_slice,
                                   tile_num=tile_num,
                                   tile_split_mode=tile_split_mode_ins)

        return weight_mapping_dict

    def get_catch_error_info_list(self):

        catch_error_info_list = []
        errer_info = "share memory bank group id is not avail"
        line_buffer_errer_info = "line_buffer_size overflow, need to split block in w axis"
        catch_error_info_list.append(errer_info)
        catch_error_info_list.append(line_buffer_errer_info)
        for bg_id in self.bank_group_id_list:
            catch_error_info = f"share memory bank group id:{bg_id} memory allocate error"
            catch_error_info_list.append(catch_error_info)
            catch_error_info_1 = f"share memory bank group id:{bg_id} already used"
            catch_error_info_list.append(catch_error_info_1)
        return catch_error_info_list

    def vpu_memory_check(self, output_tensor_info):
        # 16chnnal 下 w 不能为奇数
        block_address_list = output_tensor_info["block_address_list"]
        need_allocate = output_tensor_info.get("need_allocate", None)
        tensor_size = output_tensor_info.get("tensor_size", None)
        cl = block_address_list[5]
        wl = block_address_list[3]
        hl = block_address_list[3]
        if cl == 16:
            if wl % 2 != 0:
                assert need_allocate == True
                new_tensor_size = hl * (wl + 1) * cl
                assert new_tensor_size > tensor_size
                output_tensor_info["tensor_size"] = new_tensor_size
                return True
            else:
                return False
        else:
            return False

    def tile_address_list_check(self, tile_address_list):
        shape = tile_address_list.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if tile_address_list[i][j][k][-1] == 16 and tile_address_list[i][j][k][3] % 2 == 1:
                        tile_address_list[i][j][k][3] += 1

    def origin_shape_check(self, origin_shape):
        if origin_shape[-1] == 16 and origin_shape[1] % 2 == 1:
            origin_shape[1] += 1
