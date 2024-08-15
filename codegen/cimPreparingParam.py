from codegen.utils import *


class CimClusterRegister():
    para_psum_quant_en = 0
    para_bist_mode_en = 1
    para_clusters_bypass_mode = 2
    para_clusters_quantized_bypass = 3
    para_clusters_bias_mode = 4
    para_cims_bias_mode = 5
    para_clusters_weight_mode = 6
    para_cims_weight_mode = 7
    para_clusters_add_mode = 8
    para_cims_add_mode = 9
    para_cims_work_mode = 10
    para_clu_soft_rstn = 11
    para_cims_checkpoint_num = 12
    para_cims_acc_num = 13
    para_cims_acc_sel = 14
    para_rd_weight_b_addr = 15
    para_rd_cim0_weight_length = 16
    para_rd_cim1_weight_length = 17
    para_rd_cim2_weight_length = 18
    para_rd_cim3_weight_length = 19
    para_rd_bias_total_length = 20
    para_cims_bias_en0 = 21
    para_cims_bias_en1 = 22
    para_cims_bias_en2 = 23
    para_cims_bias_en3 = 24
    para_clusters_bias_en = 25
    para_rd_quantized_total_length = 26
    para_rstn_ckgate_tm = 27
    para_clusters_ckgate_tenable = 28
    para_clusters_ckgate_enable = 29
    para_bist_mode_done = 30
    para_weight_load_done = 31
    para_bist_mode_seed = 32
    para_bist_mode_data = 33


def get_cim_cluster_dict(cim_cluster_dict):
    para_psum_quant_en = cim_cluster_dict[CimClusterRegister.para_psum_quant_en]
    para_bist_mode_en = cim_cluster_dict[CimClusterRegister.para_bist_mode_en]
    para_clusters_bypass_mode = cim_cluster_dict[CimClusterRegister.para_clusters_bypass_mode]
    para_clusters_quantized_bypass = cim_cluster_dict[CimClusterRegister.para_clusters_quantized_bypass]
    para_clusters_bias_mode = cim_cluster_dict[CimClusterRegister.para_clusters_bias_mode]
    para_cims_bias_mode = cim_cluster_dict[CimClusterRegister.para_cims_bias_mode]
    para_clusters_weight_mode = cim_cluster_dict[CimClusterRegister.para_clusters_weight_mode]
    para_cims_weight_mode = cim_cluster_dict[CimClusterRegister.para_cims_weight_mode]
    para_clusters_add_mode = cim_cluster_dict[CimClusterRegister.para_clusters_add_mode]
    para_cims_add_mode = cim_cluster_dict[CimClusterRegister.para_cims_add_mode]
    para_cims_work_mode = cim_cluster_dict[CimClusterRegister.para_cims_work_mode]
    para_clu_soft_rstn = cim_cluster_dict[CimClusterRegister.para_clu_soft_rstn]
    para_cims_checkpoint_num = cim_cluster_dict[CimClusterRegister.para_cims_checkpoint_num]
    para_cims_acc_num = cim_cluster_dict[CimClusterRegister.para_cims_acc_num]
    para_cims_acc_sel = cim_cluster_dict[CimClusterRegister.para_cims_acc_sel]
    para_rd_weight_b_addr = cim_cluster_dict[CimClusterRegister.para_rd_weight_b_addr]
    para_rd_cim0_weight_length = cim_cluster_dict[CimClusterRegister.para_rd_cim0_weight_length]
    para_rd_cim1_weight_length = cim_cluster_dict[CimClusterRegister.para_rd_cim1_weight_length]
    para_rd_cim2_weight_length = cim_cluster_dict[CimClusterRegister.para_rd_cim2_weight_length]
    para_rd_cim3_weight_length = cim_cluster_dict[CimClusterRegister.para_rd_cim3_weight_length]
    para_rd_bias_total_length = cim_cluster_dict[CimClusterRegister.para_rd_bias_total_length]
    para_cims_bias_en0 = cim_cluster_dict[CimClusterRegister.para_cims_bias_en0]
    para_cims_bias_en1 = cim_cluster_dict[CimClusterRegister.para_cims_bias_en1]
    para_cims_bias_en2 = cim_cluster_dict[CimClusterRegister.para_cims_bias_en2]
    para_cims_bias_en3 = cim_cluster_dict[CimClusterRegister.para_cims_bias_en3]
    para_clusters_bias_en = cim_cluster_dict[CimClusterRegister.para_clusters_bias_en]
    para_rd_quantized_total_length = cim_cluster_dict[CimClusterRegister.para_rd_quantized_total_length]
    para_rstn_ckgate_tm = cim_cluster_dict[CimClusterRegister.para_rstn_ckgate_tm]
    para_clusters_ckgate_tenable = cim_cluster_dict[CimClusterRegister.para_clusters_ckgate_tenable]
    para_clusters_ckgate_enable = cim_cluster_dict[CimClusterRegister.para_clusters_ckgate_enable]
    para_bist_mode_done = cim_cluster_dict[CimClusterRegister.para_bist_mode_done]
    para_weight_load_done = cim_cluster_dict[CimClusterRegister.para_weight_load_done]
    para_bist_mode_seed = cim_cluster_dict[CimClusterRegister.para_bist_mode_seed]
    para_bist_mode_data = cim_cluster_dict[CimClusterRegister.para_bist_mode_data]

    pre_process_dict = {}
    pre_process_dict["para_psum_quant_en"] = para_psum_quant_en
    pre_process_dict["para_bist_mode_en"] = para_bist_mode_en
    pre_process_dict["para_clusters_bypass_mode"] = para_clusters_bypass_mode
    pre_process_dict["para_clusters_quantized_bypass"] = para_clusters_quantized_bypass
    pre_process_dict["para_clusters_bias_mode"] = para_clusters_bias_mode
    pre_process_dict["para_cims_bias_mode"] = para_cims_bias_mode
    pre_process_dict["para_clusters_weight_mode"] = para_clusters_weight_mode
    pre_process_dict["para_cims_weight_mode"] = para_cims_weight_mode
    pre_process_dict["para_clusters_add_mode"] = para_clusters_add_mode
    pre_process_dict["para_cims_add_mode"] = para_cims_add_mode
    pre_process_dict["para_cims_work_mode"] = para_cims_work_mode
    pre_process_dict["para_clu_soft_rstn"] = para_clu_soft_rstn
    pre_process_dict["para_cims_checkpoint_num"] = para_cims_checkpoint_num
    pre_process_dict["para_cims_acc_num"] = para_cims_acc_num
    pre_process_dict["para_cims_acc_sel"] = para_cims_acc_sel
    pre_process_dict["para_rd_weight_b_addr"] = para_rd_weight_b_addr
    pre_process_dict["para_rd_cim0_weight_length"] = para_rd_cim0_weight_length
    pre_process_dict["para_rd_cim1_weight_length"] = para_rd_cim1_weight_length
    pre_process_dict["para_rd_cim2_weight_length"] = para_rd_cim2_weight_length
    pre_process_dict["para_rd_cim3_weight_length"] = para_rd_cim3_weight_length
    pre_process_dict["para_rd_bias_total_length"] = para_rd_bias_total_length
    pre_process_dict["para_cims_bias_en0"] = para_cims_bias_en0
    pre_process_dict["para_cims_bias_en1"] = para_cims_bias_en1
    pre_process_dict["para_cims_bias_en2"] = para_cims_bias_en2
    pre_process_dict["para_cims_bias_en3"] = para_cims_bias_en3
    pre_process_dict["para_clusters_bias_en"] = para_clusters_bias_en
    pre_process_dict["para_rd_quantized_total_length"] = para_rd_quantized_total_length
    pre_process_dict["para_rstn_ckgate_tm"] = para_rstn_ckgate_tm
    pre_process_dict["para_clusters_ckgate_tenable"] = para_clusters_ckgate_tenable
    pre_process_dict["para_clusters_ckgate_enable"] = para_clusters_ckgate_enable
    pre_process_dict["para_bist_mode_done"] = para_bist_mode_done
    pre_process_dict["para_weight_load_done"] = para_weight_load_done
    pre_process_dict["para_bist_mode_seed"] = para_bist_mode_seed
    pre_process_dict["para_bist_mode_data"] = para_bist_mode_data
    return pre_process_dict


def cim_cluster_param(cim_cluster_dict):
    register_dict = []
    # CLU_PARA_CLU_COMM
    CLU_PARA_CLU_COMM = []
    CLU_PARA_CLU_COMM.append(n_zeros_str(1))
    para_psum_quant_en = cim_cluster_dict["para_psum_quant_en"]
    CLU_PARA_CLU_COMM.append(intToBin(para_psum_quant_en, 1))
    para_bist_mode_en = cim_cluster_dict["para_bist_mode_en"]
    CLU_PARA_CLU_COMM.append(intToBin(para_bist_mode_en, 1))
    para_clusters_bypass_mode = cim_cluster_dict["para_clusters_bypass_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clusters_bypass_mode, 2))
    para_clusters_quantized_bypass = cim_cluster_dict["para_clusters_quantized_bypass"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clusters_quantized_bypass, 5))
    para_clusters_bias_mode = cim_cluster_dict["para_clusters_bias_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clusters_bias_mode, 2))
    para_cims_bias_mode = cim_cluster_dict["para_cims_bias_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_bias_mode, 2))
    para_clusters_weight_mode = cim_cluster_dict["para_clusters_weight_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clusters_weight_mode, 2))
    para_cims_weight_mode = cim_cluster_dict["para_cims_weight_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_weight_mode, 2))
    para_clusters_add_mode = cim_cluster_dict["para_clusters_add_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clusters_add_mode, 2))
    para_cims_add_mode = cim_cluster_dict["para_cims_add_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_add_mode, 2))
    para_cims_work_mode = cim_cluster_dict["para_cims_work_mode"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_work_mode, 3))
    para_clu_soft_rstn = cim_cluster_dict["para_clu_soft_rstn"]
    CLU_PARA_CLU_COMM.append(intToBin(para_clu_soft_rstn, 1))
    para_cims_checkpoint_num = cim_cluster_dict["para_cims_checkpoint_num"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_checkpoint_num, 2))
    para_cims_acc_num = cim_cluster_dict["para_cims_acc_num"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_acc_num, 2))
    para_cims_acc_sel = cim_cluster_dict["para_cims_acc_sel"]
    CLU_PARA_CLU_COMM.append(intToBin(para_cims_acc_sel, 2))
    CLU_PARA_CLU_COMM = bin_listTobin(CLU_PARA_CLU_COMM)
    CLU_PARA_CLU_COMM = binTohex(CLU_PARA_CLU_COMM, 32)
    register_dict.append(CLU_PARA_CLU_COMM)
    # CLU_RD_WEIGHT_B_ADDR
    CLU_RD_WEIGHT_B_ADDR = []
    CLU_RD_WEIGHT_B_ADDR.append(n_zeros_str(13))
    para_rd_weight_b_addr = cim_cluster_dict["para_rd_weight_b_addr"]
    CLU_RD_WEIGHT_B_ADDR.append(intToBin(para_rd_weight_b_addr, 19))
    CLU_RD_WEIGHT_B_ADDR = bin_listTobin(CLU_RD_WEIGHT_B_ADDR)
    CLU_RD_WEIGHT_B_ADDR = binTohex(CLU_RD_WEIGHT_B_ADDR, 32)
    register_dict.append(CLU_RD_WEIGHT_B_ADDR)
    # CLU_RD_CIM0_WEIGHT_LENGTH
    CLU_RD_CIM0_WEIGHT_LENGTH = []
    CLU_RD_CIM0_WEIGHT_LENGTH.append(n_zeros_str(13))
    para_rd_cim0_weight_length = cim_cluster_dict["para_rd_cim0_weight_length"]
    CLU_RD_CIM0_WEIGHT_LENGTH.append(intToBin(para_rd_cim0_weight_length, 19))
    CLU_RD_CIM0_WEIGHT_LENGTH = bin_listTobin(CLU_RD_CIM0_WEIGHT_LENGTH)
    CLU_RD_CIM0_WEIGHT_LENGTH = binTohex(CLU_RD_CIM0_WEIGHT_LENGTH, 32)
    register_dict.append(CLU_RD_CIM0_WEIGHT_LENGTH)
    # CLU_RD_CIM1_WEIGHT_LENGTH
    CLU_RD_CIM1_WEIGHT_LENGTH = []
    CLU_RD_CIM1_WEIGHT_LENGTH.append(n_zeros_str(13))
    para_rd_cim1_weight_length = cim_cluster_dict["para_rd_cim1_weight_length"]
    CLU_RD_CIM1_WEIGHT_LENGTH.append(intToBin(para_rd_cim1_weight_length, 19))
    CLU_RD_CIM1_WEIGHT_LENGTH = bin_listTobin(CLU_RD_CIM1_WEIGHT_LENGTH)
    CLU_RD_CIM1_WEIGHT_LENGTH = binTohex(CLU_RD_CIM1_WEIGHT_LENGTH, 32)
    register_dict.append(CLU_RD_CIM1_WEIGHT_LENGTH)
    # CLU_RD_CIM2_WEIGHT_LENGTH
    CLU_RD_CIM2_WEIGHT_LENGTH = []
    CLU_RD_CIM2_WEIGHT_LENGTH.append(n_zeros_str(13))
    para_rd_cim2_weight_length = cim_cluster_dict["para_rd_cim2_weight_length"]
    CLU_RD_CIM2_WEIGHT_LENGTH.append(intToBin(para_rd_cim2_weight_length, 19))
    CLU_RD_CIM2_WEIGHT_LENGTH = bin_listTobin(CLU_RD_CIM2_WEIGHT_LENGTH)
    CLU_RD_CIM2_WEIGHT_LENGTH = binTohex(CLU_RD_CIM2_WEIGHT_LENGTH, 32)
    register_dict.append(CLU_RD_CIM2_WEIGHT_LENGTH)
    # CLU_RD_CIM3_WEIGHT_LENGTH
    CLU_RD_CIM3_WEIGHT_LENGTH = []
    CLU_RD_CIM3_WEIGHT_LENGTH.append(n_zeros_str(13))
    para_rd_cim3_weight_length = cim_cluster_dict["para_rd_cim3_weight_length"]
    CLU_RD_CIM3_WEIGHT_LENGTH.append(intToBin(para_rd_cim3_weight_length, 19))
    CLU_RD_CIM3_WEIGHT_LENGTH = bin_listTobin(CLU_RD_CIM3_WEIGHT_LENGTH)
    CLU_RD_CIM3_WEIGHT_LENGTH = binTohex(CLU_RD_CIM3_WEIGHT_LENGTH, 32)
    register_dict.append(CLU_RD_CIM3_WEIGHT_LENGTH)
    # CLU_RD_BIAS_TOTAL_LENGTH
    CLU_RD_BIAS_TOTAL_LENGTH = []
    CLU_RD_BIAS_TOTAL_LENGTH.append(n_zeros_str(13))
    para_rd_bias_total_length = cim_cluster_dict["para_rd_bias_total_length"]
    CLU_RD_BIAS_TOTAL_LENGTH.append(intToBin(para_rd_bias_total_length, 19))
    CLU_RD_BIAS_TOTAL_LENGTH = bin_listTobin(CLU_RD_BIAS_TOTAL_LENGTH)
    CLU_RD_BIAS_TOTAL_LENGTH = binTohex(CLU_RD_BIAS_TOTAL_LENGTH, 32)
    register_dict.append(CLU_RD_BIAS_TOTAL_LENGTH)
    # CLU_BIAS_EN_COMM
    CLU_BIAS_EN_COMM = []
    CLU_BIAS_EN_COMM.append(n_zeros_str(14))
    para_cims_bias_en0 = cim_cluster_dict["para_cims_bias_en0"]
    CLU_BIAS_EN_COMM.append(intToBin(para_cims_bias_en0, 4))
    para_cims_bias_en1 = cim_cluster_dict["para_cims_bias_en1"]
    CLU_BIAS_EN_COMM.append(intToBin(para_cims_bias_en1, 4))
    para_cims_bias_en2 = cim_cluster_dict["para_cims_bias_en2"]
    CLU_BIAS_EN_COMM.append(intToBin(para_cims_bias_en2, 4))
    para_cims_bias_en3 = cim_cluster_dict["para_cims_bias_en3"]
    CLU_BIAS_EN_COMM.append(intToBin(para_cims_bias_en3, 4))
    para_clusters_bias_en = cim_cluster_dict["para_clusters_bias_en"]
    CLU_BIAS_EN_COMM.append(intToBin(para_clusters_bias_en, 2))
    CLU_BIAS_EN_COMM = bin_listTobin(CLU_BIAS_EN_COMM)
    CLU_BIAS_EN_COMM = binTohex(CLU_BIAS_EN_COMM, 32)
    register_dict.append(CLU_BIAS_EN_COMM)
    # CLU_RD_QUANTIZED_TOTAL_LENGTH
    CLU_RD_QUANTIZED_TOTAL_LENGTH = []
    CLU_RD_QUANTIZED_TOTAL_LENGTH.append(n_zeros_str(13))
    para_rd_quantized_total_length = cim_cluster_dict["para_rd_quantized_total_length"]
    CLU_RD_QUANTIZED_TOTAL_LENGTH.append(intToBin(para_rd_quantized_total_length, 19))
    CLU_RD_QUANTIZED_TOTAL_LENGTH = bin_listTobin(CLU_RD_QUANTIZED_TOTAL_LENGTH)
    CLU_RD_QUANTIZED_TOTAL_LENGTH = binTohex(CLU_RD_QUANTIZED_TOTAL_LENGTH, 32)
    register_dict.append(CLU_RD_QUANTIZED_TOTAL_LENGTH)
    # CLU_CLUSTERS_CKGATE_ENABLE
    CLU_CLUSTERS_CKGATE_ENABLE = []
    CLU_CLUSTERS_CKGATE_ENABLE.append(n_zeros_str(23))
    para_rstn_ckgate_tm = cim_cluster_dict["para_rstn_ckgate_tm"]
    CLU_CLUSTERS_CKGATE_ENABLE.append(intToBin(para_rstn_ckgate_tm, 1))
    para_clusters_ckgate_tenable = cim_cluster_dict["para_clusters_ckgate_tenable"]
    CLU_CLUSTERS_CKGATE_ENABLE.append(intToBin(para_clusters_ckgate_tenable, 4))
    para_clusters_ckgate_enable = cim_cluster_dict["para_clusters_ckgate_enable"]
    CLU_CLUSTERS_CKGATE_ENABLE.append(intToBin(para_clusters_ckgate_enable, 4))
    CLU_CLUSTERS_CKGATE_ENABLE = bin_listTobin(CLU_CLUSTERS_CKGATE_ENABLE)
    CLU_CLUSTERS_CKGATE_ENABLE = binTohex(CLU_CLUSTERS_CKGATE_ENABLE, 32)
    register_dict.append(CLU_CLUSTERS_CKGATE_ENABLE)
    # CLU_STATE_CLU_COMM
    CLU_STATE_CLU_COMM = []
    CLU_STATE_CLU_COMM.append(n_zeros_str(30))
    para_bist_mode_done = cim_cluster_dict["para_bist_mode_done"]
    CLU_STATE_CLU_COMM.append(intToBin(para_bist_mode_done, 1))
    para_weight_load_done = cim_cluster_dict["para_weight_load_done"]
    CLU_STATE_CLU_COMM.append(intToBin(para_weight_load_done, 1))
    CLU_STATE_CLU_COMM = bin_listTobin(CLU_STATE_CLU_COMM)
    CLU_STATE_CLU_COMM = binTohex(CLU_STATE_CLU_COMM, 32)
    register_dict.append(CLU_STATE_CLU_COMM)
    # CLU_BIST_MODE_SEED
    CLU_BIST_MODE_SEED = []
    para_bist_mode_seed = cim_cluster_dict["para_bist_mode_seed"]
    CLU_BIST_MODE_SEED.append(intToBin(para_bist_mode_seed, 32))
    CLU_BIST_MODE_SEED = bin_listTobin(CLU_BIST_MODE_SEED)
    CLU_BIST_MODE_SEED = binTohex(CLU_BIST_MODE_SEED, 32)
    register_dict.append(CLU_BIST_MODE_SEED)
    # CLU_BIST_MODE_DATA
    CLU_BIST_MODE_DATA = []
    para_bist_mode_data = cim_cluster_dict["para_bist_mode_data"]
    CLU_BIST_MODE_DATA.append(intToBin(para_bist_mode_data, 32))
    CLU_BIST_MODE_DATA = bin_listTobin(CLU_BIST_MODE_DATA)
    CLU_BIST_MODE_DATA = binTohex(CLU_BIST_MODE_DATA, 32)
    register_dict.append(CLU_BIST_MODE_DATA)
    return register_dict


if __name__ == "__main__":
    para_psum_quant_en = 0
    para_bist_mode_en = 0
    para_clusters_bypass_mode = 0
    para_clusters_quantized_bypass = 0
    para_clusters_bias_mode = 0
    para_cims_bias_mode = 0
    para_clusters_weight_mode = 0
    para_cims_weight_mode = 0
    para_clusters_add_mode = 0
    para_cims_add_mode = 0
    para_cims_work_mode = 0
    para_clu_soft_rstn = 0
    para_cims_checkpoint_num = 0
    para_cims_acc_num = 0
    para_cims_acc_sel = 0
    para_rd_weight_b_addr = 0
    para_rd_cim0_weight_length = 0
    para_rd_cim1_weight_length = 0
    para_rd_cim2_weight_length = 0
    para_rd_cim3_weight_length = 0
    para_rd_bias_total_length = 0
    para_cims_bias_en0 = 0
    para_cims_bias_en1 = 0
    para_cims_bias_en2 = 0
    para_cims_bias_en3 = 0
    para_clusters_bias_en = 0
    para_rd_quantized_total_length = 0
    para_rstn_ckgate_tm = 0
    para_clusters_ckgate_tenable = 0
    para_clusters_ckgate_enable = 0
    para_bist_mode_done = 0
    para_weight_load_done = 0
    para_bist_mode_seed = 0
    para_bist_mode_data = 0
    vpu_dict = [para_psum_quant_en, para_bist_mode_en, para_clusters_bypass_mode, para_clusters_quantized_bypass,
                para_clusters_bias_mode, para_cims_bias_mode, para_clusters_weight_mode, para_cims_weight_mode,
                para_clusters_add_mode, para_cims_add_mode, para_cims_work_mode, para_clu_soft_rstn,
                para_cims_checkpoint_num, para_cims_acc_num, para_cims_acc_sel, para_rd_weight_b_addr,
                para_rd_cim0_weight_length, para_rd_cim1_weight_length, para_rd_cim2_weight_length,
                para_rd_cim3_weight_length, para_rd_bias_total_length, para_cims_bias_en0, para_cims_bias_en1,
                para_cims_bias_en2, para_cims_bias_en3, para_clusters_bias_en, para_rd_quantized_total_length,
                para_rstn_ckgate_tm, para_clusters_ckgate_tenable, para_clusters_ckgate_enable, para_bist_mode_done,
                para_weight_load_done, para_bist_mode_seed, para_bist_mode_data]
