from codegen.utils import *


class VpuRegister():
    vpu_SF_para_vpu_fifo_group_sf_rst = 0
    vpu_SF_para_global_line_buffer_sf_rst = 1
    vpu_SF_para_sc_buffer_sf_rst = 2
    vpu_SF_para_vpu_unit_sf_rst = 3
    vpu_SF_para_interface_sf_rst = 4
    vpu_SF_para_top_ctrl_sf_rst = 5
    vpu_TEST_para_done_len = 6
    vpu_TEST_para_bp_mode = 7
    vpu_TEST_para_vpu_unit_test_mode_sel = 8
    vpu_TEST_para_vpu_unit_test_mode_enable = 9
    vpu_SF_para_odd_output_enable = 10
    vpu_SF_para_bp_mode_enable = 11
    vpu_SF_para_short_st_arb_enable = 12
    vpu_SF_para_psum_enable = 13
    vpu_SF_para_short_cut_buffer = 14
    vpu_SF_para_line_controller_3 = 15
    vpu_SF_para_line_controller_2 = 16
    vpu_SF_para_line_controller_1 = 17
    vpu_SF_para_line_controller_0 = 18
    vpu_SF_para_fifo2line_buffer_enable = 19
    vpu_SF_para_global_line_buffer_enable = 20
    vpu_SF_para_input_fifo_enable = 21
    vpu_SF_para_vpu_sc_mode = 22
    vpu_SF_para_vpu_sc_enable = 23
    vpu_SF_para_vpu_enable = 24
    vpu_TOP_para_cim_weights_mode = 25
    vpu_TOP_para_cluster_weights_mode = 26
    vpu_TOP_para_interface_write_mode = 27
    vpu_TOP_para_sc_width = 28
    vpu_TOP_para_switch_mode = 29
    vpu_TOP_para_sc_mode = 30
    vpu_TOP_para_line_buffer_mode = 31
    vpu_TOP_para_fmt_channel_type = 32
    vpu_TOP_para_read_line_buffer_mode = 33
    vpu_WISE_para_quantize_min = 34
    vpu_WISE_para_quantize_max = 35
    vpu_WISE_para_mode = 36
    vpu_WISE_para_quantize_mul = 37
    vpu_WISE_para_quantize_shf = 38
    vpu_WISE_para_quantize_off = 39
    vpu_WISE_para_element_wise_dequantize_0_sclale_o = 40
    vpu_WISE_para_element_wise_dequantize_0_shifter_o = 41
    vpu_WISE_para_dequantize_0_off = 42
    vpu_WISE_para_element_wise_dequantize_1_sclale_o = 43
    vpu_WISE_para_element_wise_dequantize_1_shifter_o = 44
    vpu_WISE_para_dequantize_1_off = 45
    vpu_WISE_para_div_fix_param = 46
    vpu_WISE_para_div_shifter = 47
    vpu_BASIC_para_i_resize_param_half_pixal_flag = 48
    vpu_BASIC_para_i_resize_param_bil_nn_sel_flag = 49
    vpu_BASIC_para_pl_func_mode = 50
    vpu_BASIC_para_pl_factor = 51
    vpu_FILTER_para_i_pooling_filter_width = 52
    vpu_FILTER_para_i_pooling_filter_height = 53
    vpu_STRIDE_para_i_pooling_stride_width = 54
    vpu_STRIDE_para_i_pooling_stride_height = 55
    vpu_BASIC_para_i_fmt_width = 56
    vpu_BASIC_para_i_pl_rs_sel = 57
    vpu_INTERFACE_para_b0_ad = 58
    vpu_INTERFACE_para_b1_ad = 59
    vpu_INTERFACE_para_b2_ad = 60
    vpu_INTERFACE_para_b3_ad = 61
    vpu_INTERFACE_para_b4_wt_addr = 62
    vpu_INTERFACE_para_b5_wt_addr = 63
    vpu_INTERFACE_para_b6_wt_addr = 64
    vpu_INTERFACE_para_b7_wt_addr = 65
    vpu_INTERFACE_para_b4_rd_addr = 66
    vpu_INTERFACE_para_b5_rd_addr = 67
    vpu_INTERFACE_para_b6_rd_addr = 68
    vpu_INTERFACE_para_b7_rd_addr = 69
    vpu_GLOBAL_para_sc_buffer_stov = 70
    vpu_GLOBAL_para_sc_buffer_ema = 71
    vpu_GLOBAL_para_sc_buffer_emaw = 72
    vpu_GLOBAL_para_sc_buffer_emas = 73
    vpu_GLOBAL_para_sc_buffer_ret1n = 74
    vpu_GLOBAL_para_sc_buffer_rawl = 75
    vpu_GLOBAL_para_sc_buffer_rawlm = 76
    vpu_GLOBAL_para_sc_buffer_wabl = 77
    vpu_GLOBAL_para_sc_buffer_wablm = 78
    vpu_GLOBAL_para_global_buffer_stov = 79
    vpu_GLOBAL_para_global_buffer_ema = 80
    vpu_GLOBAL_para_global_buffer_emaw = 81
    vpu_GLOBAL_para_global_buffer_emas = 82
    vpu_GLOBAL_para_global_buffer_ret1n = 83
    vpu_GLOBAL_para_global_buffer_rawl = 84
    vpu_GLOBAL_para_global_buffer_rawlm = 85
    vpu_GLOBAL_para_global_buffer_wabl = 86
    vpu_GLOBAL_para_global_buffer_wablm = 87
    vpu_LINE_para_line_buffer_stov = 88
    vpu_LINE_para_line_buffer_ema = 89
    vpu_LINE_para_line_buffer_emaw = 90
    vpu_LINE_para_line_buffer_emas = 91
    vpu_LINE_para_line_buffer_ret1n = 92
    vpu_LINE_para_line_buffer_rawl = 93
    vpu_LINE_para_line_buffer_rawlm = 94
    vpu_LINE_para_line_buffer_wabl = 95
    vpu_LINE_para_line_buffer_wablm = 96
    vpu_LUT_para_lut_stov = 97
    vpu_LUT_para_lut_ema = 98
    vpu_LUT_para_lut_emaw = 99
    vpu_LUT_para_lut_emas = 100
    vpu_LUT_para_lut_ret1n = 101
    vpu_LUT_para_lut_rawl = 102
    vpu_LUT_para_lut_rawlm = 103
    vpu_LUT_para_lut_wabl = 104
    vpu_LUT_para_lut_wablm = 105
    vpu_UNIT_para_i_vpu_unit_seed = 106
    vpu_INTERFACE_para_i_vpu_interface_seed = 107
    vpu_IN_para_i_vpu_in_fifo_group_seed = 108
    vpu_DATA_para_i_target_data_amount = 109
    vpu_BIST_para_o_ans_data = 110
    vpu_BIST_para_o_ans_data_sc = 111
    vpu_BIST_para_o_ans_data_wo_q = 112
    vpu_BIST_para_o_ans_done_flag = 113
    vpu_BIST_para_o_ans_col_done = 114
    vpu_BIST_para_all_ans_data_out_psum = 115
    vpu_BIST_para_all_ans_data_out_0 = 116
    vpu_BIST_para_all_ans_data_out_1 = 117
    vpu_BIST_para_all_ans_data_out_2 = 118
    vpu_BIST_para_all_ans_data_out_3 = 119
    vpu_BIST_para_o_b0_ans_data = 120
    vpu_BIST_para_o_b0_ans_ctrl = 121
    vpu_BIST_para_o_b0_ans_addr = 122
    vpu_BIST_para_o_b1_ans_data = 123
    vpu_BIST_para_o_b1_ans_ctrl = 124
    vpu_BIST_para_o_b1_ans_addr = 125
    vpu_BIST_para_o_b2_ans_data = 126
    vpu_BIST_para_o_b2_ans_ctrl = 127
    vpu_BIST_para_o_b2_ans_addr = 128
    vpu_BIST_para_o_b3_ans_data = 129
    vpu_BIST_para_o_b3_ans_ctrl = 130
    vpu_BIST_para_o_b3_ans_addr = 131
    vpu_BIST_para_o_b4_ans_data = 132
    vpu_BIST_para_o_b4_ans_ctrl = 133
    vpu_BIST_para_o_b4_ans_addr = 134
    vpu_BIST_para_o_b5_ans_data = 135
    vpu_BIST_para_o_b5_ans_ctrl = 136
    vpu_BIST_para_o_b5_ans_addr = 137
    vpu_BIST_para_o_b6_ans_data = 138
    vpu_BIST_para_o_b6_ans_ctrl = 139
    vpu_BIST_para_o_b6_ans_addr = 140
    vpu_BIST_para_o_b7_ans_data = 141
    vpu_BIST_para_o_b7_ans_ctrl = 142
    vpu_BIST_para_o_b7_ans_addr = 143
    vpu_0_para_clus_0_scw_ad_0 = 144
    vpu_0_para_clus_0_scr_ad_0 = 145
    vpu_0_para_clus_0_ad_0 = 146
    vpu_0_para_clus_0_scw_ad_1 = 147
    vpu_0_para_clus_0_scr_ad_1 = 148
    vpu_0_para_clus_0_ad_1 = 149
    vpu_0_para_clus_0_scw_ad_2 = 150
    vpu_0_para_clus_0_scr_ad_2 = 151
    vpu_0_para_clus_0_ad_2 = 152
    vpu_0_para_clus_0_scw_ad_3 = 153
    vpu_0_para_clus_0_scr_ad_3 = 154
    vpu_0_para_clus_0_ad_3 = 155
    vpu_0_para_clus_0_block_ad_jump_0 = 156
    vpu_0_para_clus_0_block_ad_mode_enable = 157
    vpu_0_para_clus_0_block_ad_jump_condit1 = 158
    vpu_0_para_clus_0_block_ad_jump_condit0 = 159
    vpu_0_para_clus_0_block_ad_jump_2 = 160
    vpu_0_para_clus_0_block_ad_jump_1 = 161
    vpu_0_para_clus_0_block_scr_ad_jump_0 = 162
    vpu_0_para_clus_0_block_scr_ad_jump_condit1 = 163
    vpu_0_para_clus_0_block_scr_ad_jump_condit0 = 164
    vpu_0_para_clus_0_block_scr_ad_jump_2 = 165
    vpu_0_para_clus_0_block_scr_ad_jump_1 = 166
    vpu_0_para_clus_0_block_scw_ad_jump_0 = 167
    vpu_0_para_clus_0_block_scw_ad_jump_condit1 = 168
    vpu_0_para_clus_0_block_scw_ad_jump_condit0 = 169
    vpu_0_para_clus_0_block_scw_ad_jump_2 = 170
    vpu_0_para_clus_0_block_scw_ad_jump_1 = 171
    vpu_0_para_clus_0_line_buffer_w_max = 172
    vpu_0_para_clus_0_line_buffer_h_max = 173
    vpu_0_para_clus_0_kernal_h = 174
    vpu_0_para_clus_0_kernal_h_stride = 175
    vpu_0_para_clus_0_output_l1_step = 176
    vpu_0_para_clus_0_output_l1_condition = 177
    vpu_0_para_clus_0_output_l2_step = 178
    vpu_0_para_clus_0_output_l2_condition = 179
    vpu_0_para_clus_0_output_l3_step = 180
    vpu_0_para_clus_0_output_l3_condition = 181
    vpu_0_para_clus_0_output_l1_addr_step = 182
    vpu_0_para_clus_0_output_l2_addr_step = 183
    vpu_0_para_clus_0_output_l3_addr_step = 184
    vpu_0_para_clus_0_scw_l1_step = 185
    vpu_0_para_clus_0_scw_l1_condition = 186
    vpu_0_para_clus_0_scw_l2_step = 187
    vpu_0_para_clus_0_scw_l2_condition = 188
    vpu_0_para_clus_0_scw_l3_step = 189
    vpu_0_para_clus_0_scw_l3_condition = 190
    vpu_0_para_clus_0_scw_l1_addr_step = 191
    vpu_0_para_clus_0_scw_l2_addr_step = 192
    vpu_0_para_clus_0_scw_l3_addr_step = 193
    vpu_0_para_clus_0_scr_l1_step = 194
    vpu_0_para_clus_0_scr_l1_condition = 195
    vpu_0_para_clus_0_scr_l2_step = 196
    vpu_0_para_clus_0_scr_l2_condition = 197
    vpu_0_para_clus_0_scr_l3_step = 198
    vpu_0_para_clus_0_scr_l3_condition = 199
    vpu_0_para_clus_0_scr_l1_addr_step = 200
    vpu_0_para_clus_0_scr_l2_addr_step = 201
    vpu_0_para_clus_0_scr_l3_addr_step = 202
    vpu_0_para_i_clus_0_resize_param_width_ratio = 203
    vpu_0_para_i_clus_0_resize_param_height_ratio = 204
    vpu_0_para_i_clus_0_resize_param_input_width = 205
    vpu_0_para_i_clus_0_resize_param_input_height = 206
    vpu_0_para_i_clus_0_resize_param_output_width = 207
    vpu_0_para_i_clus_0_resize_param_output_height = 208
    vpu_0_para_i_clus_0_pooling_param_input_width = 209
    vpu_0_para_i_clus_0_pooling_param_input_height = 210
    vpu_0_para_i_clus_0_pooling_param_output_width = 211
    vpu_0_para_i_clus_0_pooling_param_output_height = 212
    vpu_0_para_i_clus_0_pooling_padding_mode = 213
    vpu_0_para_i_clus_0_pooling_padding_width = 214
    vpu_0_para_i_clus_0_pooling_padding_height = 215
    vpu_1_para_clus_1_scw_ad_0 = 216
    vpu_1_para_clus_1_scr_ad_0 = 217
    vpu_1_para_clus_1_ad_0 = 218
    vpu_1_para_clus_1_scw_ad_1 = 219
    vpu_1_para_clus_1_scr_ad_1 = 220
    vpu_1_para_clus_1_ad_1 = 221
    vpu_1_para_clus_1_scw_ad_2 = 222
    vpu_1_para_clus_1_scr_ad_2 = 223
    vpu_1_para_clus_1_ad_2 = 224
    vpu_1_para_clus_1_scw_ad_3 = 225
    vpu_1_para_clus_1_scr_ad_3 = 226
    vpu_1_para_clus_1_ad_3 = 227
    vpu_1_para_clus_1_block_ad_jump_0 = 228
    vpu_1_para_clus_1_block_ad_mode_enable = 229
    vpu_1_para_clus_1_block_ad_jump_condit1 = 230
    vpu_1_para_clus_1_block_ad_jump_condit0 = 231
    vpu_1_para_clus_1_block_ad_jump_2 = 232
    vpu_1_para_clus_1_block_ad_jump_1 = 233
    vpu_1_para_clus_1_block_scr_ad_jump_0 = 234
    vpu_1_para_clus_1_block_scr_ad_jump_condit1 = 235
    vpu_1_para_clus_1_block_scr_ad_jump_condit0 = 236
    vpu_1_para_clus_1_block_scr_ad_jump_2 = 237
    vpu_1_para_clus_1_block_scr_ad_jump_1 = 238
    vpu_1_para_clus_1_block_scw_ad_jump_0 = 239
    vpu_1_para_clus_1_block_scw_ad_jump_condit1 = 240
    vpu_1_para_clus_1_block_scw_ad_jump_condit0 = 241
    vpu_1_para_clus_1_block_scw_ad_jump_2 = 242
    vpu_1_para_clus_1_block_scw_ad_jump_1 = 243
    vpu_1_para_clus_1_line_buffer_w_max = 244
    vpu_1_para_clus_1_line_buffer_h_max = 245
    vpu_1_para_clus_1_kernal_h = 246
    vpu_1_para_clus_1_kernal_h_stride = 247
    vpu_1_para_clus_1_output_l1_step = 248
    vpu_1_para_clus_1_output_l1_condition = 249
    vpu_1_para_clus_1_output_l2_step = 250
    vpu_1_para_clus_1_output_l2_condition = 251
    vpu_1_para_clus_1_output_l3_step = 252
    vpu_1_para_clus_1_output_l3_condition = 253
    vpu_1_para_clus_1_output_l1_addr_step = 254
    vpu_1_para_clus_1_output_l2_addr_step = 255
    vpu_1_para_clus_1_output_l3_addr_step = 256
    vpu_1_para_clus_1_scw_l1_step = 257
    vpu_1_para_clus_1_scw_l1_condition = 258
    vpu_1_para_clus_1_scw_l2_step = 259
    vpu_1_para_clus_1_scw_l2_condition = 260
    vpu_1_para_clus_1_scw_l3_step = 261
    vpu_1_para_clus_1_scw_l3_condition = 262
    vpu_1_para_clus_1_scw_l1_addr_step = 263
    vpu_1_para_clus_1_scw_l2_addr_step = 264
    vpu_1_para_clus_1_scw_l3_addr_step = 265
    vpu_1_para_clus_1_scr_l1_step = 266
    vpu_1_para_clus_1_scr_l1_condition = 267
    vpu_1_para_clus_1_scr_l2_step = 268
    vpu_1_para_clus_1_scr_l2_condition = 269
    vpu_1_para_clus_1_scr_l3_step = 270
    vpu_1_para_clus_1_scr_l3_condition = 271
    vpu_1_para_clus_1_scr_l1_addr_step = 272
    vpu_1_para_clus_1_scr_l2_addr_step = 273
    vpu_1_para_clus_1_scr_l3_addr_step = 274
    vpu_1_para_i_clus_1_resize_param_width_ratio = 275
    vpu_1_para_i_clus_1_resize_param_height_ratio = 276
    vpu_1_para_i_clus_1_resize_param_input_width = 277
    vpu_1_para_i_clus_1_resize_param_input_height = 278
    vpu_1_para_i_clus_1_resize_param_output_width = 279
    vpu_1_para_i_clus_1_resize_param_output_height = 280
    vpu_1_para_i_clus_1_pooling_param_input_width = 281
    vpu_1_para_i_clus_1_pooling_param_input_height = 282
    vpu_1_para_i_clus_1_pooling_param_output_width = 283
    vpu_1_para_i_clus_1_pooling_param_output_height = 284
    vpu_1_para_i_clus_1_pooling_padding_mode = 285
    vpu_1_para_i_clus_1_pooling_padding_width = 286
    vpu_1_para_i_clus_1_pooling_padding_height = 287
    vpu_2_para_clus_2_scw_ad_0 = 288
    vpu_2_para_clus_2_scr_ad_0 = 289
    vpu_2_para_clus_2_ad_0 = 290
    vpu_2_para_clus_2_scw_ad_1 = 291
    vpu_2_para_clus_2_scr_ad_1 = 292
    vpu_2_para_clus_2_ad_1 = 293
    vpu_2_para_clus_2_scw_ad_2 = 294
    vpu_2_para_clus_2_scr_ad_2 = 295
    vpu_2_para_clus_2_ad_2 = 296
    vpu_2_para_clus_2_scw_ad_3 = 297
    vpu_2_para_clus_2_scr_ad_3 = 298
    vpu_2_para_clus_2_ad_3 = 299
    vpu_2_para_clus_2_block_ad_jump_0 = 300
    vpu_2_para_clus_2_block_ad_mode_enable = 301
    vpu_2_para_clus_2_block_ad_jump_condit1 = 302
    vpu_2_para_clus_2_block_ad_jump_condit0 = 303
    vpu_2_para_clus_2_block_ad_jump_2 = 304
    vpu_2_para_clus_2_block_ad_jump_1 = 305
    vpu_2_para_clus_2_block_scr_ad_jump_0 = 306
    vpu_2_para_clus_2_block_scr_ad_jump_condit1 = 307
    vpu_2_para_clus_2_block_scr_ad_jump_condit0 = 308
    vpu_2_para_clus_2_block_scr_ad_jump_2 = 309
    vpu_2_para_clus_2_block_scr_ad_jump_1 = 310
    vpu_2_para_clus_2_block_scw_ad_jump_0 = 311
    vpu_2_para_clus_2_block_scw_ad_jump_condit1 = 312
    vpu_2_para_clus_2_block_scw_ad_jump_condit0 = 313
    vpu_2_para_clus_2_block_scw_ad_jump_2 = 314
    vpu_2_para_clus_2_block_scw_ad_jump_1 = 315
    vpu_2_para_clus_2_line_buffer_w_max = 316
    vpu_2_para_clus_2_line_buffer_h_max = 317
    vpu_2_para_clus_2_kernal_h = 318
    vpu_2_para_clus_2_kernal_h_stride = 319
    vpu_2_para_clus_2_output_l1_step = 320
    vpu_2_para_clus_2_output_l1_condition = 321
    vpu_2_para_clus_2_output_l2_step = 322
    vpu_2_para_clus_2_output_l2_condition = 323
    vpu_2_para_clus_2_output_l3_step = 324
    vpu_2_para_clus_2_output_l3_condition = 325
    vpu_2_para_clus_2_output_l1_addr_step = 326
    vpu_2_para_clus_2_output_l2_addr_step = 327
    vpu_2_para_clus_2_output_l3_addr_step = 328
    vpu_2_para_clus_2_scw_l1_step = 329
    vpu_2_para_clus_2_scw_l1_condition = 330
    vpu_2_para_clus_2_scw_l2_step = 331
    vpu_2_para_clus_2_scw_l2_condition = 332
    vpu_2_para_clus_2_scw_l3_step = 333
    vpu_2_para_clus_2_scw_l3_condition = 334
    vpu_2_para_clus_2_scw_l1_addr_step = 335
    vpu_2_para_clus_2_scw_l2_addr_step = 336
    vpu_2_para_clus_2_scw_l3_addr_step = 337
    vpu_2_para_clus_2_scr_l1_step = 338
    vpu_2_para_clus_2_scr_l1_condition = 339
    vpu_2_para_clus_2_scr_l2_step = 340
    vpu_2_para_clus_2_scr_l2_condition = 341
    vpu_2_para_clus_2_scr_l3_step = 342
    vpu_2_para_clus_2_scr_l3_condition = 343
    vpu_2_para_clus_2_scr_l1_addr_step = 344
    vpu_2_para_clus_2_scr_l2_addr_step = 345
    vpu_2_para_clus_2_scr_l3_addr_step = 346
    vpu_2_para_i_clus_2_resize_param_width_ratio = 347
    vpu_2_para_i_clus_2_resize_param_height_ratio = 348
    vpu_2_para_i_clus_2_resize_param_input_width = 349
    vpu_2_para_i_clus_2_resize_param_input_height = 350
    vpu_2_para_i_clus_2_resize_param_output_width = 351
    vpu_2_para_i_clus_2_resize_param_output_height = 352
    vpu_2_para_i_clus_2_pooling_param_input_width = 353
    vpu_2_para_i_clus_2_pooling_param_input_height = 354
    vpu_2_para_i_clus_2_pooling_param_output_width = 355
    vpu_2_para_i_clus_2_pooling_param_output_height = 356
    vpu_2_para_i_clus_2_pooling_padding_mode = 357
    vpu_2_para_i_clus_2_pooling_padding_width = 358
    vpu_2_para_i_clus_2_pooling_padding_height = 359
    vpu_3_para_clus_3_scw_ad_0 = 360
    vpu_3_para_clus_3_scr_ad_0 = 361
    vpu_3_para_clus_3_ad_0 = 362
    vpu_3_para_clus_3_scw_ad_1 = 363
    vpu_3_para_clus_3_scr_ad_1 = 364
    vpu_3_para_clus_3_ad_1 = 365
    vpu_3_para_clus_3_scw_ad_2 = 366
    vpu_3_para_clus_3_scr_ad_2 = 367
    vpu_3_para_clus_3_ad_2 = 368
    vpu_3_para_clus_3_scw_ad_3 = 369
    vpu_3_para_clus_3_scr_ad_3 = 370
    vpu_3_para_clus_3_ad_3 = 371
    vpu_3_para_clus_3_block_ad_jump_0 = 372
    vpu_3_para_clus_3_block_ad_mode_enable = 373
    vpu_3_para_clus_3_block_ad_jump_condit1 = 374
    vpu_3_para_clus_3_block_ad_jump_condit0 = 375
    vpu_3_para_clus_3_block_ad_jump_2 = 376
    vpu_3_para_clus_3_block_ad_jump_1 = 377
    vpu_3_para_clus_3_block_scr_ad_jump_0 = 378
    vpu_3_para_clus_3_block_scr_ad_jump_condit1 = 379
    vpu_3_para_clus_3_block_scr_ad_jump_condit0 = 380
    vpu_3_para_clus_3_block_scr_ad_jump_2 = 381
    vpu_3_para_clus_3_block_scr_ad_jump_1 = 382
    vpu_3_para_clus_3_block_scw_ad_jump_0 = 383
    vpu_3_para_clus_3_block_scw_ad_jump_condit1 = 384
    vpu_3_para_clus_3_block_scw_ad_jump_condit0 = 385
    vpu_3_para_clus_3_block_scw_ad_jump_2 = 386
    vpu_3_para_clus_3_block_scw_ad_jump_1 = 387
    vpu_3_para_clus_3_line_buffer_w_max = 388
    vpu_3_para_clus_3_line_buffer_h_max = 389
    vpu_3_para_clus_3_kernal_h = 390
    vpu_3_para_clus_3_kernal_h_stride = 391
    vpu_3_para_clus_3_output_l1_step = 392
    vpu_3_para_clus_3_output_l1_condition = 393
    vpu_3_para_clus_3_output_l2_step = 394
    vpu_3_para_clus_3_output_l2_condition = 395
    vpu_3_para_clus_3_output_l3_step = 396
    vpu_3_para_clus_3_output_l3_condition = 397
    vpu_3_para_clus_3_output_l1_addr_step = 398
    vpu_3_para_clus_3_output_l2_addr_step = 399
    vpu_3_para_clus_3_output_l3_addr_step = 400
    vpu_3_para_clus_3_scw_l1_step = 401
    vpu_3_para_clus_3_scw_l1_condition = 402
    vpu_3_para_clus_3_scw_l2_step = 403
    vpu_3_para_clus_3_scw_l2_condition = 404
    vpu_3_para_clus_3_scw_l3_step = 405
    vpu_3_para_clus_3_scw_l3_condition = 406
    vpu_3_para_clus_3_scw_l1_addr_step = 407
    vpu_3_para_clus_3_scw_l2_addr_step = 408
    vpu_3_para_clus_3_scw_l3_addr_step = 409
    vpu_3_para_clus_3_scr_l1_step = 410
    vpu_3_para_clus_3_scr_l1_condition = 411
    vpu_3_para_clus_3_scr_l2_step = 412
    vpu_3_para_clus_3_scr_l2_condition = 413
    vpu_3_para_clus_3_scr_l3_step = 414
    vpu_3_para_clus_3_scr_l3_condition = 415
    vpu_3_para_clus_3_scr_l1_addr_step = 416
    vpu_3_para_clus_3_scr_l2_addr_step = 417
    vpu_3_para_clus_3_scr_l3_addr_step = 418
    vpu_3_para_i_clus_3_resize_param_width_ratio = 419
    vpu_3_para_i_clus_3_resize_param_height_ratio = 420
    vpu_3_para_i_clus_3_resize_param_input_width = 421
    vpu_3_para_i_clus_3_resize_param_input_height = 422
    vpu_3_para_i_clus_3_resize_param_output_width = 423
    vpu_3_para_i_clus_3_resize_param_output_height = 424
    vpu_3_para_i_clus_3_pooling_param_input_width = 425
    vpu_3_para_i_clus_3_pooling_param_input_height = 426
    vpu_3_para_i_clus_3_pooling_param_output_width = 427
    vpu_3_para_i_clus_3_pooling_param_output_height = 428
    vpu_3_para_i_clus_3_pooling_padding_mode = 429
    vpu_3_para_i_clus_3_pooling_padding_width = 430
    vpu_3_para_i_clus_3_pooling_padding_height = 431


def get_vpu_dict(vpu_dict):
    vpu_SF_para_vpu_fifo_group_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_vpu_fifo_group_sf_rst]
    vpu_SF_para_global_line_buffer_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_global_line_buffer_sf_rst]
    vpu_SF_para_sc_buffer_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_sc_buffer_sf_rst]
    vpu_SF_para_vpu_unit_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_vpu_unit_sf_rst]
    vpu_SF_para_interface_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_interface_sf_rst]
    vpu_SF_para_top_ctrl_sf_rst = vpu_dict[VpuRegister.vpu_SF_para_top_ctrl_sf_rst]
    vpu_TEST_para_done_len = vpu_dict[VpuRegister.vpu_TEST_para_done_len]
    vpu_TEST_para_bp_mode = vpu_dict[VpuRegister.vpu_TEST_para_bp_mode]
    vpu_TEST_para_vpu_unit_test_mode_sel = vpu_dict[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_sel]
    vpu_TEST_para_vpu_unit_test_mode_enable = vpu_dict[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_enable]
    vpu_SF_para_odd_output_enable = vpu_dict[VpuRegister.vpu_SF_para_odd_output_enable]
    vpu_SF_para_bp_mode_enable = vpu_dict[VpuRegister.vpu_SF_para_bp_mode_enable]
    vpu_SF_para_short_st_arb_enable = vpu_dict[VpuRegister.vpu_SF_para_short_st_arb_enable]
    vpu_SF_para_psum_enable = vpu_dict[VpuRegister.vpu_SF_para_psum_enable]
    vpu_SF_para_short_cut_buffer = vpu_dict[VpuRegister.vpu_SF_para_short_cut_buffer]
    vpu_SF_para_line_controller_3 = vpu_dict[VpuRegister.vpu_SF_para_line_controller_3]
    vpu_SF_para_line_controller_2 = vpu_dict[VpuRegister.vpu_SF_para_line_controller_2]
    vpu_SF_para_line_controller_1 = vpu_dict[VpuRegister.vpu_SF_para_line_controller_1]
    vpu_SF_para_line_controller_0 = vpu_dict[VpuRegister.vpu_SF_para_line_controller_0]
    vpu_SF_para_fifo2line_buffer_enable = vpu_dict[VpuRegister.vpu_SF_para_fifo2line_buffer_enable]
    vpu_SF_para_global_line_buffer_enable = vpu_dict[VpuRegister.vpu_SF_para_global_line_buffer_enable]
    vpu_SF_para_input_fifo_enable = vpu_dict[VpuRegister.vpu_SF_para_input_fifo_enable]
    vpu_SF_para_vpu_sc_mode = vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_mode]
    vpu_SF_para_vpu_sc_enable = vpu_dict[VpuRegister.vpu_SF_para_vpu_sc_enable]
    vpu_SF_para_vpu_enable = vpu_dict[VpuRegister.vpu_SF_para_vpu_enable]
    vpu_TOP_para_cim_weights_mode = vpu_dict[VpuRegister.vpu_TOP_para_cim_weights_mode]
    vpu_TOP_para_cluster_weights_mode = vpu_dict[VpuRegister.vpu_TOP_para_cluster_weights_mode]
    vpu_TOP_para_interface_write_mode = vpu_dict[VpuRegister.vpu_TOP_para_interface_write_mode]
    vpu_TOP_para_sc_width = vpu_dict[VpuRegister.vpu_TOP_para_sc_width]
    vpu_TOP_para_switch_mode = vpu_dict[VpuRegister.vpu_TOP_para_switch_mode]
    vpu_TOP_para_sc_mode = vpu_dict[VpuRegister.vpu_TOP_para_sc_mode]
    vpu_TOP_para_line_buffer_mode = vpu_dict[VpuRegister.vpu_TOP_para_line_buffer_mode]
    vpu_TOP_para_fmt_channel_type = vpu_dict[VpuRegister.vpu_TOP_para_fmt_channel_type]
    vpu_TOP_para_read_line_buffer_mode = vpu_dict[VpuRegister.vpu_TOP_para_read_line_buffer_mode]
    vpu_WISE_para_quantize_min = vpu_dict[VpuRegister.vpu_WISE_para_quantize_min]
    vpu_WISE_para_quantize_max = vpu_dict[VpuRegister.vpu_WISE_para_quantize_max]
    vpu_WISE_para_mode = vpu_dict[VpuRegister.vpu_WISE_para_mode]
    vpu_WISE_para_quantize_mul = vpu_dict[VpuRegister.vpu_WISE_para_quantize_mul]
    vpu_WISE_para_quantize_shf = vpu_dict[VpuRegister.vpu_WISE_para_quantize_shf]
    vpu_WISE_para_quantize_off = vpu_dict[VpuRegister.vpu_WISE_para_quantize_off]
    vpu_WISE_para_element_wise_dequantize_0_sclale_o = vpu_dict[
        VpuRegister.vpu_WISE_para_element_wise_dequantize_0_sclale_o]
    vpu_WISE_para_element_wise_dequantize_0_shifter_o = vpu_dict[
        VpuRegister.vpu_WISE_para_element_wise_dequantize_0_shifter_o]
    vpu_WISE_para_dequantize_0_off = vpu_dict[VpuRegister.vpu_WISE_para_dequantize_0_off]
    vpu_WISE_para_element_wise_dequantize_1_sclale_o = vpu_dict[
        VpuRegister.vpu_WISE_para_element_wise_dequantize_1_sclale_o]
    vpu_WISE_para_element_wise_dequantize_1_shifter_o = vpu_dict[
        VpuRegister.vpu_WISE_para_element_wise_dequantize_1_shifter_o]
    vpu_WISE_para_dequantize_1_off = vpu_dict[VpuRegister.vpu_WISE_para_dequantize_1_off]
    vpu_WISE_para_div_fix_param = vpu_dict[VpuRegister.vpu_WISE_para_div_fix_param]
    vpu_WISE_para_div_shifter = vpu_dict[VpuRegister.vpu_WISE_para_div_shifter]
    vpu_BASIC_para_i_resize_param_half_pixal_flag = vpu_dict[VpuRegister.vpu_BASIC_para_i_resize_param_half_pixal_flag]
    vpu_BASIC_para_i_resize_param_bil_nn_sel_flag = vpu_dict[VpuRegister.vpu_BASIC_para_i_resize_param_bil_nn_sel_flag]
    vpu_BASIC_para_pl_func_mode = vpu_dict[VpuRegister.vpu_BASIC_para_pl_func_mode]
    vpu_BASIC_para_pl_factor = vpu_dict[VpuRegister.vpu_BASIC_para_pl_factor]
    vpu_FILTER_para_i_pooling_filter_width = vpu_dict[VpuRegister.vpu_FILTER_para_i_pooling_filter_width]
    vpu_FILTER_para_i_pooling_filter_height = vpu_dict[VpuRegister.vpu_FILTER_para_i_pooling_filter_height]
    vpu_STRIDE_para_i_pooling_stride_width = vpu_dict[VpuRegister.vpu_STRIDE_para_i_pooling_stride_width]
    vpu_STRIDE_para_i_pooling_stride_height = vpu_dict[VpuRegister.vpu_STRIDE_para_i_pooling_stride_height]
    vpu_BASIC_para_i_fmt_width = vpu_dict[VpuRegister.vpu_BASIC_para_i_fmt_width]
    vpu_BASIC_para_i_pl_rs_sel = vpu_dict[VpuRegister.vpu_BASIC_para_i_pl_rs_sel]
    vpu_INTERFACE_para_b0_ad = vpu_dict[VpuRegister.vpu_INTERFACE_para_b0_ad]
    vpu_INTERFACE_para_b1_ad = vpu_dict[VpuRegister.vpu_INTERFACE_para_b1_ad]
    vpu_INTERFACE_para_b2_ad = vpu_dict[VpuRegister.vpu_INTERFACE_para_b2_ad]
    vpu_INTERFACE_para_b3_ad = vpu_dict[VpuRegister.vpu_INTERFACE_para_b3_ad]
    vpu_INTERFACE_para_b4_wt_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b4_wt_addr]
    vpu_INTERFACE_para_b5_wt_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b5_wt_addr]
    vpu_INTERFACE_para_b6_wt_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b6_wt_addr]
    vpu_INTERFACE_para_b7_wt_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b7_wt_addr]
    vpu_INTERFACE_para_b4_rd_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b4_rd_addr]
    vpu_INTERFACE_para_b5_rd_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b5_rd_addr]
    vpu_INTERFACE_para_b6_rd_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b6_rd_addr]
    vpu_INTERFACE_para_b7_rd_addr = vpu_dict[VpuRegister.vpu_INTERFACE_para_b7_rd_addr]
    vpu_GLOBAL_para_sc_buffer_stov = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_stov]
    vpu_GLOBAL_para_sc_buffer_ema = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_ema]
    vpu_GLOBAL_para_sc_buffer_emaw = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_emaw]
    vpu_GLOBAL_para_sc_buffer_emas = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_emas]
    vpu_GLOBAL_para_sc_buffer_ret1n = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_ret1n]
    vpu_GLOBAL_para_sc_buffer_rawl = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawl]
    vpu_GLOBAL_para_sc_buffer_rawlm = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawlm]
    vpu_GLOBAL_para_sc_buffer_wabl = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_wabl]
    vpu_GLOBAL_para_sc_buffer_wablm = vpu_dict[VpuRegister.vpu_GLOBAL_para_sc_buffer_wablm]
    vpu_GLOBAL_para_global_buffer_stov = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_stov]
    vpu_GLOBAL_para_global_buffer_ema = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_ema]
    vpu_GLOBAL_para_global_buffer_emaw = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_emaw]
    vpu_GLOBAL_para_global_buffer_emas = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_emas]
    vpu_GLOBAL_para_global_buffer_ret1n = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_ret1n]
    vpu_GLOBAL_para_global_buffer_rawl = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_rawl]
    vpu_GLOBAL_para_global_buffer_rawlm = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_rawlm]
    vpu_GLOBAL_para_global_buffer_wabl = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_wabl]
    vpu_GLOBAL_para_global_buffer_wablm = vpu_dict[VpuRegister.vpu_GLOBAL_para_global_buffer_wablm]
    vpu_LINE_para_line_buffer_stov = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_stov]
    vpu_LINE_para_line_buffer_ema = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_ema]
    vpu_LINE_para_line_buffer_emaw = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_emaw]
    vpu_LINE_para_line_buffer_emas = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_emas]
    vpu_LINE_para_line_buffer_ret1n = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_ret1n]
    vpu_LINE_para_line_buffer_rawl = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_rawl]
    vpu_LINE_para_line_buffer_rawlm = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_rawlm]
    vpu_LINE_para_line_buffer_wabl = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_wabl]
    vpu_LINE_para_line_buffer_wablm = vpu_dict[VpuRegister.vpu_LINE_para_line_buffer_wablm]
    vpu_LUT_para_lut_stov = vpu_dict[VpuRegister.vpu_LUT_para_lut_stov]
    vpu_LUT_para_lut_ema = vpu_dict[VpuRegister.vpu_LUT_para_lut_ema]
    vpu_LUT_para_lut_emaw = vpu_dict[VpuRegister.vpu_LUT_para_lut_emaw]
    vpu_LUT_para_lut_emas = vpu_dict[VpuRegister.vpu_LUT_para_lut_emas]
    vpu_LUT_para_lut_ret1n = vpu_dict[VpuRegister.vpu_LUT_para_lut_ret1n]
    vpu_LUT_para_lut_rawl = vpu_dict[VpuRegister.vpu_LUT_para_lut_rawl]
    vpu_LUT_para_lut_rawlm = vpu_dict[VpuRegister.vpu_LUT_para_lut_rawlm]
    vpu_LUT_para_lut_wabl = vpu_dict[VpuRegister.vpu_LUT_para_lut_wabl]
    vpu_LUT_para_lut_wablm = vpu_dict[VpuRegister.vpu_LUT_para_lut_wablm]
    vpu_UNIT_para_i_vpu_unit_seed = vpu_dict[VpuRegister.vpu_UNIT_para_i_vpu_unit_seed]
    vpu_INTERFACE_para_i_vpu_interface_seed = vpu_dict[VpuRegister.vpu_INTERFACE_para_i_vpu_interface_seed]
    vpu_IN_para_i_vpu_in_fifo_group_seed = vpu_dict[VpuRegister.vpu_IN_para_i_vpu_in_fifo_group_seed]
    vpu_DATA_para_i_target_data_amount = vpu_dict[VpuRegister.vpu_DATA_para_i_target_data_amount]
    vpu_BIST_para_o_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_ans_data]
    vpu_BIST_para_o_ans_data_sc = vpu_dict[VpuRegister.vpu_BIST_para_o_ans_data_sc]
    vpu_BIST_para_o_ans_data_wo_q = vpu_dict[VpuRegister.vpu_BIST_para_o_ans_data_wo_q]
    vpu_BIST_para_o_ans_done_flag = vpu_dict[VpuRegister.vpu_BIST_para_o_ans_done_flag]
    vpu_BIST_para_o_ans_col_done = vpu_dict[VpuRegister.vpu_BIST_para_o_ans_col_done]
    vpu_BIST_para_all_ans_data_out_psum = vpu_dict[VpuRegister.vpu_BIST_para_all_ans_data_out_psum]
    vpu_BIST_para_all_ans_data_out_0 = vpu_dict[VpuRegister.vpu_BIST_para_all_ans_data_out_0]
    vpu_BIST_para_all_ans_data_out_1 = vpu_dict[VpuRegister.vpu_BIST_para_all_ans_data_out_1]
    vpu_BIST_para_all_ans_data_out_2 = vpu_dict[VpuRegister.vpu_BIST_para_all_ans_data_out_2]
    vpu_BIST_para_all_ans_data_out_3 = vpu_dict[VpuRegister.vpu_BIST_para_all_ans_data_out_3]
    vpu_BIST_para_o_b0_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b0_ans_data]
    vpu_BIST_para_o_b0_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b0_ans_ctrl]
    vpu_BIST_para_o_b0_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b0_ans_addr]
    vpu_BIST_para_o_b1_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b1_ans_data]
    vpu_BIST_para_o_b1_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b1_ans_ctrl]
    vpu_BIST_para_o_b1_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b1_ans_addr]
    vpu_BIST_para_o_b2_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b2_ans_data]
    vpu_BIST_para_o_b2_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b2_ans_ctrl]
    vpu_BIST_para_o_b2_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b2_ans_addr]
    vpu_BIST_para_o_b3_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b3_ans_data]
    vpu_BIST_para_o_b3_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b3_ans_ctrl]
    vpu_BIST_para_o_b3_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b3_ans_addr]
    vpu_BIST_para_o_b4_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b4_ans_data]
    vpu_BIST_para_o_b4_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b4_ans_ctrl]
    vpu_BIST_para_o_b4_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b4_ans_addr]
    vpu_BIST_para_o_b5_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b5_ans_data]
    vpu_BIST_para_o_b5_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b5_ans_ctrl]
    vpu_BIST_para_o_b5_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b5_ans_addr]
    vpu_BIST_para_o_b6_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b6_ans_data]
    vpu_BIST_para_o_b6_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b6_ans_ctrl]
    vpu_BIST_para_o_b6_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b6_ans_addr]
    vpu_BIST_para_o_b7_ans_data = vpu_dict[VpuRegister.vpu_BIST_para_o_b7_ans_data]
    vpu_BIST_para_o_b7_ans_ctrl = vpu_dict[VpuRegister.vpu_BIST_para_o_b7_ans_ctrl]
    vpu_BIST_para_o_b7_ans_addr = vpu_dict[VpuRegister.vpu_BIST_para_o_b7_ans_addr]
    vpu_0_para_clus_0_scw_ad_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_0]
    vpu_0_para_clus_0_scr_ad_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_0]
    vpu_0_para_clus_0_ad_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_0]
    vpu_0_para_clus_0_scw_ad_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_1]
    vpu_0_para_clus_0_scr_ad_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_1]
    vpu_0_para_clus_0_ad_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_1]
    vpu_0_para_clus_0_scw_ad_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_2]
    vpu_0_para_clus_0_scr_ad_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_2]
    vpu_0_para_clus_0_ad_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_2]
    vpu_0_para_clus_0_scw_ad_3 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_ad_3]
    vpu_0_para_clus_0_scr_ad_3 = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_ad_3]
    vpu_0_para_clus_0_ad_3 = vpu_dict[VpuRegister.vpu_0_para_clus_0_ad_3]
    vpu_0_para_clus_0_block_ad_jump_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_jump_0]
    vpu_0_para_clus_0_block_ad_mode_enable = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_mode_enable]
    vpu_0_para_clus_0_block_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_jump_condit1]
    vpu_0_para_clus_0_block_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_jump_condit0]
    vpu_0_para_clus_0_block_ad_jump_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_jump_2]
    vpu_0_para_clus_0_block_ad_jump_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_ad_jump_1]
    vpu_0_para_clus_0_block_scr_ad_jump_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_0]
    vpu_0_para_clus_0_block_scr_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_condit1]
    vpu_0_para_clus_0_block_scr_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_condit0]
    vpu_0_para_clus_0_block_scr_ad_jump_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_2]
    vpu_0_para_clus_0_block_scr_ad_jump_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_1]
    vpu_0_para_clus_0_block_scw_ad_jump_0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_0]
    vpu_0_para_clus_0_block_scw_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_condit1]
    vpu_0_para_clus_0_block_scw_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_condit0]
    vpu_0_para_clus_0_block_scw_ad_jump_2 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_2]
    vpu_0_para_clus_0_block_scw_ad_jump_1 = vpu_dict[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_1]
    vpu_0_para_clus_0_line_buffer_w_max = vpu_dict[VpuRegister.vpu_0_para_clus_0_line_buffer_w_max]
    vpu_0_para_clus_0_line_buffer_h_max = vpu_dict[VpuRegister.vpu_0_para_clus_0_line_buffer_h_max]
    vpu_0_para_clus_0_kernal_h = vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h]
    vpu_0_para_clus_0_kernal_h_stride = vpu_dict[VpuRegister.vpu_0_para_clus_0_kernal_h_stride]
    vpu_0_para_clus_0_output_l1_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l1_step]
    vpu_0_para_clus_0_output_l1_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l1_condition]
    vpu_0_para_clus_0_output_l2_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l2_step]
    vpu_0_para_clus_0_output_l2_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l2_condition]
    vpu_0_para_clus_0_output_l3_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l3_step]
    vpu_0_para_clus_0_output_l3_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l3_condition]
    vpu_0_para_clus_0_output_l1_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l1_addr_step]
    vpu_0_para_clus_0_output_l2_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l2_addr_step]
    vpu_0_para_clus_0_output_l3_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_output_l3_addr_step]
    vpu_0_para_clus_0_scw_l1_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l1_step]
    vpu_0_para_clus_0_scw_l1_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l1_condition]
    vpu_0_para_clus_0_scw_l2_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l2_step]
    vpu_0_para_clus_0_scw_l2_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l2_condition]
    vpu_0_para_clus_0_scw_l3_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l3_step]
    vpu_0_para_clus_0_scw_l3_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l3_condition]
    vpu_0_para_clus_0_scw_l1_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l1_addr_step]
    vpu_0_para_clus_0_scw_l2_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l2_addr_step]
    vpu_0_para_clus_0_scw_l3_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scw_l3_addr_step]
    vpu_0_para_clus_0_scr_l1_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l1_step]
    vpu_0_para_clus_0_scr_l1_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l1_condition]
    vpu_0_para_clus_0_scr_l2_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l2_step]
    vpu_0_para_clus_0_scr_l2_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l2_condition]
    vpu_0_para_clus_0_scr_l3_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l3_step]
    vpu_0_para_clus_0_scr_l3_condition = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l3_condition]
    vpu_0_para_clus_0_scr_l1_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l1_addr_step]
    vpu_0_para_clus_0_scr_l2_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l2_addr_step]
    vpu_0_para_clus_0_scr_l3_addr_step = vpu_dict[VpuRegister.vpu_0_para_clus_0_scr_l3_addr_step]
    vpu_0_para_i_clus_0_resize_param_width_ratio = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_width_ratio]
    vpu_0_para_i_clus_0_resize_param_height_ratio = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_height_ratio]
    vpu_0_para_i_clus_0_resize_param_input_width = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_width]
    vpu_0_para_i_clus_0_resize_param_input_height = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_height]
    vpu_0_para_i_clus_0_resize_param_output_width = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_resize_param_output_width]
    vpu_0_para_i_clus_0_resize_param_output_height = vpu_dict[
        VpuRegister.vpu_0_para_i_clus_0_resize_param_output_height]
    vpu_0_para_i_clus_0_pooling_param_input_width = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_width]
    vpu_0_para_i_clus_0_pooling_param_input_height = vpu_dict[
        VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_height]
    vpu_0_para_i_clus_0_pooling_param_output_width = vpu_dict[
        VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_width]
    vpu_0_para_i_clus_0_pooling_param_output_height = vpu_dict[
        VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_height]
    vpu_0_para_i_clus_0_pooling_padding_mode = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_mode]
    vpu_0_para_i_clus_0_pooling_padding_width = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_width]
    vpu_0_para_i_clus_0_pooling_padding_height = vpu_dict[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_height]
    vpu_1_para_clus_1_scw_ad_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_0]
    vpu_1_para_clus_1_scr_ad_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_0]
    vpu_1_para_clus_1_ad_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_0]
    vpu_1_para_clus_1_scw_ad_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_1]
    vpu_1_para_clus_1_scr_ad_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_1]
    vpu_1_para_clus_1_ad_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_1]
    vpu_1_para_clus_1_scw_ad_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_2]
    vpu_1_para_clus_1_scr_ad_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_2]
    vpu_1_para_clus_1_ad_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_2]
    vpu_1_para_clus_1_scw_ad_3 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_ad_3]
    vpu_1_para_clus_1_scr_ad_3 = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_ad_3]
    vpu_1_para_clus_1_ad_3 = vpu_dict[VpuRegister.vpu_1_para_clus_1_ad_3]
    vpu_1_para_clus_1_block_ad_jump_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_jump_0]
    vpu_1_para_clus_1_block_ad_mode_enable = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_mode_enable]
    vpu_1_para_clus_1_block_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_jump_condit1]
    vpu_1_para_clus_1_block_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_jump_condit0]
    vpu_1_para_clus_1_block_ad_jump_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_jump_2]
    vpu_1_para_clus_1_block_ad_jump_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_ad_jump_1]
    vpu_1_para_clus_1_block_scr_ad_jump_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_0]
    vpu_1_para_clus_1_block_scr_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_condit1]
    vpu_1_para_clus_1_block_scr_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_condit0]
    vpu_1_para_clus_1_block_scr_ad_jump_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_2]
    vpu_1_para_clus_1_block_scr_ad_jump_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_1]
    vpu_1_para_clus_1_block_scw_ad_jump_0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_0]
    vpu_1_para_clus_1_block_scw_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_condit1]
    vpu_1_para_clus_1_block_scw_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_condit0]
    vpu_1_para_clus_1_block_scw_ad_jump_2 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_2]
    vpu_1_para_clus_1_block_scw_ad_jump_1 = vpu_dict[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_1]
    vpu_1_para_clus_1_line_buffer_w_max = vpu_dict[VpuRegister.vpu_1_para_clus_1_line_buffer_w_max]
    vpu_1_para_clus_1_line_buffer_h_max = vpu_dict[VpuRegister.vpu_1_para_clus_1_line_buffer_h_max]
    vpu_1_para_clus_1_kernal_h = vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h]
    vpu_1_para_clus_1_kernal_h_stride = vpu_dict[VpuRegister.vpu_1_para_clus_1_kernal_h_stride]
    vpu_1_para_clus_1_output_l1_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l1_step]
    vpu_1_para_clus_1_output_l1_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l1_condition]
    vpu_1_para_clus_1_output_l2_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l2_step]
    vpu_1_para_clus_1_output_l2_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l2_condition]
    vpu_1_para_clus_1_output_l3_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l3_step]
    vpu_1_para_clus_1_output_l3_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l3_condition]
    vpu_1_para_clus_1_output_l1_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l1_addr_step]
    vpu_1_para_clus_1_output_l2_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l2_addr_step]
    vpu_1_para_clus_1_output_l3_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_output_l3_addr_step]
    vpu_1_para_clus_1_scw_l1_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l1_step]
    vpu_1_para_clus_1_scw_l1_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l1_condition]
    vpu_1_para_clus_1_scw_l2_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l2_step]
    vpu_1_para_clus_1_scw_l2_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l2_condition]
    vpu_1_para_clus_1_scw_l3_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l3_step]
    vpu_1_para_clus_1_scw_l3_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l3_condition]
    vpu_1_para_clus_1_scw_l1_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l1_addr_step]
    vpu_1_para_clus_1_scw_l2_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l2_addr_step]
    vpu_1_para_clus_1_scw_l3_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scw_l3_addr_step]
    vpu_1_para_clus_1_scr_l1_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l1_step]
    vpu_1_para_clus_1_scr_l1_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l1_condition]
    vpu_1_para_clus_1_scr_l2_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l2_step]
    vpu_1_para_clus_1_scr_l2_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l2_condition]
    vpu_1_para_clus_1_scr_l3_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l3_step]
    vpu_1_para_clus_1_scr_l3_condition = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l3_condition]
    vpu_1_para_clus_1_scr_l1_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l1_addr_step]
    vpu_1_para_clus_1_scr_l2_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l2_addr_step]
    vpu_1_para_clus_1_scr_l3_addr_step = vpu_dict[VpuRegister.vpu_1_para_clus_1_scr_l3_addr_step]
    vpu_1_para_i_clus_1_resize_param_width_ratio = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_width_ratio]
    vpu_1_para_i_clus_1_resize_param_height_ratio = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_height_ratio]
    vpu_1_para_i_clus_1_resize_param_input_width = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_width]
    vpu_1_para_i_clus_1_resize_param_input_height = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_height]
    vpu_1_para_i_clus_1_resize_param_output_width = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_resize_param_output_width]
    vpu_1_para_i_clus_1_resize_param_output_height = vpu_dict[
        VpuRegister.vpu_1_para_i_clus_1_resize_param_output_height]
    vpu_1_para_i_clus_1_pooling_param_input_width = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_width]
    vpu_1_para_i_clus_1_pooling_param_input_height = vpu_dict[
        VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_height]
    vpu_1_para_i_clus_1_pooling_param_output_width = vpu_dict[
        VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_width]
    vpu_1_para_i_clus_1_pooling_param_output_height = vpu_dict[
        VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_height]
    vpu_1_para_i_clus_1_pooling_padding_mode = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_mode]
    vpu_1_para_i_clus_1_pooling_padding_width = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_width]
    vpu_1_para_i_clus_1_pooling_padding_height = vpu_dict[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_height]
    vpu_2_para_clus_2_scw_ad_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_0]
    vpu_2_para_clus_2_scr_ad_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_0]
    vpu_2_para_clus_2_ad_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_0]
    vpu_2_para_clus_2_scw_ad_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_1]
    vpu_2_para_clus_2_scr_ad_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_1]
    vpu_2_para_clus_2_ad_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_1]
    vpu_2_para_clus_2_scw_ad_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_2]
    vpu_2_para_clus_2_scr_ad_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_2]
    vpu_2_para_clus_2_ad_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_2]
    vpu_2_para_clus_2_scw_ad_3 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_ad_3]
    vpu_2_para_clus_2_scr_ad_3 = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_ad_3]
    vpu_2_para_clus_2_ad_3 = vpu_dict[VpuRegister.vpu_2_para_clus_2_ad_3]
    vpu_2_para_clus_2_block_ad_jump_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_jump_0]
    vpu_2_para_clus_2_block_ad_mode_enable = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_mode_enable]
    vpu_2_para_clus_2_block_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_jump_condit1]
    vpu_2_para_clus_2_block_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_jump_condit0]
    vpu_2_para_clus_2_block_ad_jump_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_jump_2]
    vpu_2_para_clus_2_block_ad_jump_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_ad_jump_1]
    vpu_2_para_clus_2_block_scr_ad_jump_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_0]
    vpu_2_para_clus_2_block_scr_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_condit1]
    vpu_2_para_clus_2_block_scr_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_condit0]
    vpu_2_para_clus_2_block_scr_ad_jump_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_2]
    vpu_2_para_clus_2_block_scr_ad_jump_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_1]
    vpu_2_para_clus_2_block_scw_ad_jump_0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_0]
    vpu_2_para_clus_2_block_scw_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_condit1]
    vpu_2_para_clus_2_block_scw_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_condit0]
    vpu_2_para_clus_2_block_scw_ad_jump_2 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_2]
    vpu_2_para_clus_2_block_scw_ad_jump_1 = vpu_dict[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_1]
    vpu_2_para_clus_2_line_buffer_w_max = vpu_dict[VpuRegister.vpu_2_para_clus_2_line_buffer_w_max]
    vpu_2_para_clus_2_line_buffer_h_max = vpu_dict[VpuRegister.vpu_2_para_clus_2_line_buffer_h_max]
    vpu_2_para_clus_2_kernal_h = vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h]
    vpu_2_para_clus_2_kernal_h_stride = vpu_dict[VpuRegister.vpu_2_para_clus_2_kernal_h_stride]
    vpu_2_para_clus_2_output_l1_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l1_step]
    vpu_2_para_clus_2_output_l1_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l1_condition]
    vpu_2_para_clus_2_output_l2_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l2_step]
    vpu_2_para_clus_2_output_l2_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l2_condition]
    vpu_2_para_clus_2_output_l3_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l3_step]
    vpu_2_para_clus_2_output_l3_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l3_condition]
    vpu_2_para_clus_2_output_l1_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l1_addr_step]
    vpu_2_para_clus_2_output_l2_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l2_addr_step]
    vpu_2_para_clus_2_output_l3_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_output_l3_addr_step]
    vpu_2_para_clus_2_scw_l1_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l1_step]
    vpu_2_para_clus_2_scw_l1_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l1_condition]
    vpu_2_para_clus_2_scw_l2_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l2_step]
    vpu_2_para_clus_2_scw_l2_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l2_condition]
    vpu_2_para_clus_2_scw_l3_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l3_step]
    vpu_2_para_clus_2_scw_l3_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l3_condition]
    vpu_2_para_clus_2_scw_l1_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l1_addr_step]
    vpu_2_para_clus_2_scw_l2_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l2_addr_step]
    vpu_2_para_clus_2_scw_l3_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scw_l3_addr_step]
    vpu_2_para_clus_2_scr_l1_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l1_step]
    vpu_2_para_clus_2_scr_l1_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l1_condition]
    vpu_2_para_clus_2_scr_l2_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l2_step]
    vpu_2_para_clus_2_scr_l2_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l2_condition]
    vpu_2_para_clus_2_scr_l3_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l3_step]
    vpu_2_para_clus_2_scr_l3_condition = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l3_condition]
    vpu_2_para_clus_2_scr_l1_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l1_addr_step]
    vpu_2_para_clus_2_scr_l2_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l2_addr_step]
    vpu_2_para_clus_2_scr_l3_addr_step = vpu_dict[VpuRegister.vpu_2_para_clus_2_scr_l3_addr_step]
    vpu_2_para_i_clus_2_resize_param_width_ratio = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_width_ratio]
    vpu_2_para_i_clus_2_resize_param_height_ratio = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_height_ratio]
    vpu_2_para_i_clus_2_resize_param_input_width = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_width]
    vpu_2_para_i_clus_2_resize_param_input_height = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_height]
    vpu_2_para_i_clus_2_resize_param_output_width = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_resize_param_output_width]
    vpu_2_para_i_clus_2_resize_param_output_height = vpu_dict[
        VpuRegister.vpu_2_para_i_clus_2_resize_param_output_height]
    vpu_2_para_i_clus_2_pooling_param_input_width = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_width]
    vpu_2_para_i_clus_2_pooling_param_input_height = vpu_dict[
        VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_height]
    vpu_2_para_i_clus_2_pooling_param_output_width = vpu_dict[
        VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_width]
    vpu_2_para_i_clus_2_pooling_param_output_height = vpu_dict[
        VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_height]
    vpu_2_para_i_clus_2_pooling_padding_mode = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_mode]
    vpu_2_para_i_clus_2_pooling_padding_width = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_width]
    vpu_2_para_i_clus_2_pooling_padding_height = vpu_dict[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_height]
    vpu_3_para_clus_3_scw_ad_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_0]
    vpu_3_para_clus_3_scr_ad_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_0]
    vpu_3_para_clus_3_ad_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_0]
    vpu_3_para_clus_3_scw_ad_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_1]
    vpu_3_para_clus_3_scr_ad_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_1]
    vpu_3_para_clus_3_ad_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_1]
    vpu_3_para_clus_3_scw_ad_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_2]
    vpu_3_para_clus_3_scr_ad_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_2]
    vpu_3_para_clus_3_ad_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_2]
    vpu_3_para_clus_3_scw_ad_3 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_ad_3]
    vpu_3_para_clus_3_scr_ad_3 = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_ad_3]
    vpu_3_para_clus_3_ad_3 = vpu_dict[VpuRegister.vpu_3_para_clus_3_ad_3]
    vpu_3_para_clus_3_block_ad_jump_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_jump_0]
    vpu_3_para_clus_3_block_ad_mode_enable = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_mode_enable]
    vpu_3_para_clus_3_block_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_jump_condit1]
    vpu_3_para_clus_3_block_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_jump_condit0]
    vpu_3_para_clus_3_block_ad_jump_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_jump_2]
    vpu_3_para_clus_3_block_ad_jump_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_ad_jump_1]
    vpu_3_para_clus_3_block_scr_ad_jump_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_0]
    vpu_3_para_clus_3_block_scr_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_condit1]
    vpu_3_para_clus_3_block_scr_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_condit0]
    vpu_3_para_clus_3_block_scr_ad_jump_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_2]
    vpu_3_para_clus_3_block_scr_ad_jump_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_1]
    vpu_3_para_clus_3_block_scw_ad_jump_0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_0]
    vpu_3_para_clus_3_block_scw_ad_jump_condit1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_condit1]
    vpu_3_para_clus_3_block_scw_ad_jump_condit0 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_condit0]
    vpu_3_para_clus_3_block_scw_ad_jump_2 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_2]
    vpu_3_para_clus_3_block_scw_ad_jump_1 = vpu_dict[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_1]
    vpu_3_para_clus_3_line_buffer_w_max = vpu_dict[VpuRegister.vpu_3_para_clus_3_line_buffer_w_max]
    vpu_3_para_clus_3_line_buffer_h_max = vpu_dict[VpuRegister.vpu_3_para_clus_3_line_buffer_h_max]
    vpu_3_para_clus_3_kernal_h = vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h]
    vpu_3_para_clus_3_kernal_h_stride = vpu_dict[VpuRegister.vpu_3_para_clus_3_kernal_h_stride]
    vpu_3_para_clus_3_output_l1_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l1_step]
    vpu_3_para_clus_3_output_l1_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l1_condition]
    vpu_3_para_clus_3_output_l2_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l2_step]
    vpu_3_para_clus_3_output_l2_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l2_condition]
    vpu_3_para_clus_3_output_l3_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l3_step]
    vpu_3_para_clus_3_output_l3_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l3_condition]
    vpu_3_para_clus_3_output_l1_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l1_addr_step]
    vpu_3_para_clus_3_output_l2_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l2_addr_step]
    vpu_3_para_clus_3_output_l3_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_output_l3_addr_step]
    vpu_3_para_clus_3_scw_l1_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l1_step]
    vpu_3_para_clus_3_scw_l1_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l1_condition]
    vpu_3_para_clus_3_scw_l2_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l2_step]
    vpu_3_para_clus_3_scw_l2_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l2_condition]
    vpu_3_para_clus_3_scw_l3_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l3_step]
    vpu_3_para_clus_3_scw_l3_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l3_condition]
    vpu_3_para_clus_3_scw_l1_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l1_addr_step]
    vpu_3_para_clus_3_scw_l2_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l2_addr_step]
    vpu_3_para_clus_3_scw_l3_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scw_l3_addr_step]
    vpu_3_para_clus_3_scr_l1_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l1_step]
    vpu_3_para_clus_3_scr_l1_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l1_condition]
    vpu_3_para_clus_3_scr_l2_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l2_step]
    vpu_3_para_clus_3_scr_l2_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l2_condition]
    vpu_3_para_clus_3_scr_l3_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l3_step]
    vpu_3_para_clus_3_scr_l3_condition = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l3_condition]
    vpu_3_para_clus_3_scr_l1_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l1_addr_step]
    vpu_3_para_clus_3_scr_l2_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l2_addr_step]
    vpu_3_para_clus_3_scr_l3_addr_step = vpu_dict[VpuRegister.vpu_3_para_clus_3_scr_l3_addr_step]
    vpu_3_para_i_clus_3_resize_param_width_ratio = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_width_ratio]
    vpu_3_para_i_clus_3_resize_param_height_ratio = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_height_ratio]
    vpu_3_para_i_clus_3_resize_param_input_width = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_width]
    vpu_3_para_i_clus_3_resize_param_input_height = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_height]
    vpu_3_para_i_clus_3_resize_param_output_width = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_resize_param_output_width]
    vpu_3_para_i_clus_3_resize_param_output_height = vpu_dict[
        VpuRegister.vpu_3_para_i_clus_3_resize_param_output_height]
    vpu_3_para_i_clus_3_pooling_param_input_width = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_width]
    vpu_3_para_i_clus_3_pooling_param_input_height = vpu_dict[
        VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_height]
    vpu_3_para_i_clus_3_pooling_param_output_width = vpu_dict[
        VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_width]
    vpu_3_para_i_clus_3_pooling_param_output_height = vpu_dict[
        VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_height]
    vpu_3_para_i_clus_3_pooling_padding_mode = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_mode]
    vpu_3_para_i_clus_3_pooling_padding_width = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_width]
    vpu_3_para_i_clus_3_pooling_padding_height = vpu_dict[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_height]

    pre_process_dict = {}
    pre_process_dict["vpu_SF_para_vpu_fifo_group_sf_rst"] = vpu_SF_para_vpu_fifo_group_sf_rst
    pre_process_dict["vpu_SF_para_global_line_buffer_sf_rst"] = vpu_SF_para_global_line_buffer_sf_rst
    pre_process_dict["vpu_SF_para_sc_buffer_sf_rst"] = vpu_SF_para_sc_buffer_sf_rst
    pre_process_dict["vpu_SF_para_vpu_unit_sf_rst"] = vpu_SF_para_vpu_unit_sf_rst
    pre_process_dict["vpu_SF_para_interface_sf_rst"] = vpu_SF_para_interface_sf_rst
    pre_process_dict["vpu_SF_para_top_ctrl_sf_rst"] = vpu_SF_para_top_ctrl_sf_rst
    pre_process_dict["vpu_TEST_para_done_len"] = vpu_TEST_para_done_len
    pre_process_dict["vpu_TEST_para_bp_mode"] = vpu_TEST_para_bp_mode
    pre_process_dict["vpu_TEST_para_vpu_unit_test_mode_sel"] = vpu_TEST_para_vpu_unit_test_mode_sel
    pre_process_dict["vpu_TEST_para_vpu_unit_test_mode_enable"] = vpu_TEST_para_vpu_unit_test_mode_enable
    pre_process_dict["vpu_SF_para_odd_output_enable"] = vpu_SF_para_odd_output_enable
    pre_process_dict["vpu_SF_para_bp_mode_enable"] = vpu_SF_para_bp_mode_enable
    pre_process_dict["vpu_SF_para_short_st_arb_enable"] = vpu_SF_para_short_st_arb_enable
    pre_process_dict["vpu_SF_para_psum_enable"] = vpu_SF_para_psum_enable
    pre_process_dict["vpu_SF_para_short_cut_buffer"] = vpu_SF_para_short_cut_buffer
    pre_process_dict["vpu_SF_para_line_controller_3"] = vpu_SF_para_line_controller_3
    pre_process_dict["vpu_SF_para_line_controller_2"] = vpu_SF_para_line_controller_2
    pre_process_dict["vpu_SF_para_line_controller_1"] = vpu_SF_para_line_controller_1
    pre_process_dict["vpu_SF_para_line_controller_0"] = vpu_SF_para_line_controller_0
    pre_process_dict["vpu_SF_para_fifo2line_buffer_enable"] = vpu_SF_para_fifo2line_buffer_enable
    pre_process_dict["vpu_SF_para_global_line_buffer_enable"] = vpu_SF_para_global_line_buffer_enable
    pre_process_dict["vpu_SF_para_input_fifo_enable"] = vpu_SF_para_input_fifo_enable
    pre_process_dict["vpu_SF_para_vpu_sc_mode"] = vpu_SF_para_vpu_sc_mode
    pre_process_dict["vpu_SF_para_vpu_sc_enable"] = vpu_SF_para_vpu_sc_enable
    pre_process_dict["vpu_SF_para_vpu_enable"] = vpu_SF_para_vpu_enable
    pre_process_dict["vpu_TOP_para_cim_weights_mode"] = vpu_TOP_para_cim_weights_mode
    pre_process_dict["vpu_TOP_para_cluster_weights_mode"] = vpu_TOP_para_cluster_weights_mode
    pre_process_dict["vpu_TOP_para_interface_write_mode"] = vpu_TOP_para_interface_write_mode
    pre_process_dict["vpu_TOP_para_sc_width"] = vpu_TOP_para_sc_width
    pre_process_dict["vpu_TOP_para_switch_mode"] = vpu_TOP_para_switch_mode
    pre_process_dict["vpu_TOP_para_sc_mode"] = vpu_TOP_para_sc_mode
    pre_process_dict["vpu_TOP_para_line_buffer_mode"] = vpu_TOP_para_line_buffer_mode
    pre_process_dict["vpu_TOP_para_fmt_channel_type"] = vpu_TOP_para_fmt_channel_type
    pre_process_dict["vpu_TOP_para_read_line_buffer_mode"] = vpu_TOP_para_read_line_buffer_mode
    pre_process_dict["vpu_WISE_para_quantize_min"] = vpu_WISE_para_quantize_min
    pre_process_dict["vpu_WISE_para_quantize_max"] = vpu_WISE_para_quantize_max
    pre_process_dict["vpu_WISE_para_mode"] = vpu_WISE_para_mode
    pre_process_dict["vpu_WISE_para_quantize_mul"] = vpu_WISE_para_quantize_mul
    pre_process_dict["vpu_WISE_para_quantize_shf"] = vpu_WISE_para_quantize_shf
    pre_process_dict["vpu_WISE_para_quantize_off"] = vpu_WISE_para_quantize_off
    pre_process_dict[
        "vpu_WISE_para_element_wise_dequantize_0_sclale_o"] = vpu_WISE_para_element_wise_dequantize_0_sclale_o
    pre_process_dict[
        "vpu_WISE_para_element_wise_dequantize_0_shifter_o"] = vpu_WISE_para_element_wise_dequantize_0_shifter_o
    pre_process_dict["vpu_WISE_para_dequantize_0_off"] = vpu_WISE_para_dequantize_0_off
    pre_process_dict[
        "vpu_WISE_para_element_wise_dequantize_1_sclale_o"] = vpu_WISE_para_element_wise_dequantize_1_sclale_o
    pre_process_dict[
        "vpu_WISE_para_element_wise_dequantize_1_shifter_o"] = vpu_WISE_para_element_wise_dequantize_1_shifter_o
    pre_process_dict["vpu_WISE_para_dequantize_1_off"] = vpu_WISE_para_dequantize_1_off
    pre_process_dict["vpu_WISE_para_div_fix_param"] = vpu_WISE_para_div_fix_param
    pre_process_dict["vpu_WISE_para_div_shifter"] = vpu_WISE_para_div_shifter
    pre_process_dict["vpu_BASIC_para_i_resize_param_half_pixal_flag"] = vpu_BASIC_para_i_resize_param_half_pixal_flag
    pre_process_dict["vpu_BASIC_para_i_resize_param_bil_nn_sel_flag"] = vpu_BASIC_para_i_resize_param_bil_nn_sel_flag
    pre_process_dict["vpu_BASIC_para_pl_func_mode"] = vpu_BASIC_para_pl_func_mode
    pre_process_dict["vpu_BASIC_para_pl_factor"] = vpu_BASIC_para_pl_factor
    pre_process_dict["vpu_FILTER_para_i_pooling_filter_width"] = vpu_FILTER_para_i_pooling_filter_width
    pre_process_dict["vpu_FILTER_para_i_pooling_filter_height"] = vpu_FILTER_para_i_pooling_filter_height
    pre_process_dict["vpu_STRIDE_para_i_pooling_stride_width"] = vpu_STRIDE_para_i_pooling_stride_width
    pre_process_dict["vpu_STRIDE_para_i_pooling_stride_height"] = vpu_STRIDE_para_i_pooling_stride_height
    pre_process_dict["vpu_BASIC_para_i_fmt_width"] = vpu_BASIC_para_i_fmt_width
    pre_process_dict["vpu_BASIC_para_i_pl_rs_sel"] = vpu_BASIC_para_i_pl_rs_sel
    pre_process_dict["vpu_INTERFACE_para_b0_ad"] = vpu_INTERFACE_para_b0_ad
    pre_process_dict["vpu_INTERFACE_para_b1_ad"] = vpu_INTERFACE_para_b1_ad
    pre_process_dict["vpu_INTERFACE_para_b2_ad"] = vpu_INTERFACE_para_b2_ad
    pre_process_dict["vpu_INTERFACE_para_b3_ad"] = vpu_INTERFACE_para_b3_ad
    pre_process_dict["vpu_INTERFACE_para_b4_wt_addr"] = vpu_INTERFACE_para_b4_wt_addr
    pre_process_dict["vpu_INTERFACE_para_b5_wt_addr"] = vpu_INTERFACE_para_b5_wt_addr
    pre_process_dict["vpu_INTERFACE_para_b6_wt_addr"] = vpu_INTERFACE_para_b6_wt_addr
    pre_process_dict["vpu_INTERFACE_para_b7_wt_addr"] = vpu_INTERFACE_para_b7_wt_addr
    pre_process_dict["vpu_INTERFACE_para_b4_rd_addr"] = vpu_INTERFACE_para_b4_rd_addr
    pre_process_dict["vpu_INTERFACE_para_b5_rd_addr"] = vpu_INTERFACE_para_b5_rd_addr
    pre_process_dict["vpu_INTERFACE_para_b6_rd_addr"] = vpu_INTERFACE_para_b6_rd_addr
    pre_process_dict["vpu_INTERFACE_para_b7_rd_addr"] = vpu_INTERFACE_para_b7_rd_addr
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_stov"] = vpu_GLOBAL_para_sc_buffer_stov
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_ema"] = vpu_GLOBAL_para_sc_buffer_ema
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_emaw"] = vpu_GLOBAL_para_sc_buffer_emaw
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_emas"] = vpu_GLOBAL_para_sc_buffer_emas
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_ret1n"] = vpu_GLOBAL_para_sc_buffer_ret1n
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_rawl"] = vpu_GLOBAL_para_sc_buffer_rawl
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_rawlm"] = vpu_GLOBAL_para_sc_buffer_rawlm
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_wabl"] = vpu_GLOBAL_para_sc_buffer_wabl
    pre_process_dict["vpu_GLOBAL_para_sc_buffer_wablm"] = vpu_GLOBAL_para_sc_buffer_wablm
    pre_process_dict["vpu_GLOBAL_para_global_buffer_stov"] = vpu_GLOBAL_para_global_buffer_stov
    pre_process_dict["vpu_GLOBAL_para_global_buffer_ema"] = vpu_GLOBAL_para_global_buffer_ema
    pre_process_dict["vpu_GLOBAL_para_global_buffer_emaw"] = vpu_GLOBAL_para_global_buffer_emaw
    pre_process_dict["vpu_GLOBAL_para_global_buffer_emas"] = vpu_GLOBAL_para_global_buffer_emas
    pre_process_dict["vpu_GLOBAL_para_global_buffer_ret1n"] = vpu_GLOBAL_para_global_buffer_ret1n
    pre_process_dict["vpu_GLOBAL_para_global_buffer_rawl"] = vpu_GLOBAL_para_global_buffer_rawl
    pre_process_dict["vpu_GLOBAL_para_global_buffer_rawlm"] = vpu_GLOBAL_para_global_buffer_rawlm
    pre_process_dict["vpu_GLOBAL_para_global_buffer_wabl"] = vpu_GLOBAL_para_global_buffer_wabl
    pre_process_dict["vpu_GLOBAL_para_global_buffer_wablm"] = vpu_GLOBAL_para_global_buffer_wablm
    pre_process_dict["vpu_LINE_para_line_buffer_stov"] = vpu_LINE_para_line_buffer_stov
    pre_process_dict["vpu_LINE_para_line_buffer_ema"] = vpu_LINE_para_line_buffer_ema
    pre_process_dict["vpu_LINE_para_line_buffer_emaw"] = vpu_LINE_para_line_buffer_emaw
    pre_process_dict["vpu_LINE_para_line_buffer_emas"] = vpu_LINE_para_line_buffer_emas
    pre_process_dict["vpu_LINE_para_line_buffer_ret1n"] = vpu_LINE_para_line_buffer_ret1n
    pre_process_dict["vpu_LINE_para_line_buffer_rawl"] = vpu_LINE_para_line_buffer_rawl
    pre_process_dict["vpu_LINE_para_line_buffer_rawlm"] = vpu_LINE_para_line_buffer_rawlm
    pre_process_dict["vpu_LINE_para_line_buffer_wabl"] = vpu_LINE_para_line_buffer_wabl
    pre_process_dict["vpu_LINE_para_line_buffer_wablm"] = vpu_LINE_para_line_buffer_wablm
    pre_process_dict["vpu_LUT_para_lut_stov"] = vpu_LUT_para_lut_stov
    pre_process_dict["vpu_LUT_para_lut_ema"] = vpu_LUT_para_lut_ema
    pre_process_dict["vpu_LUT_para_lut_emaw"] = vpu_LUT_para_lut_emaw
    pre_process_dict["vpu_LUT_para_lut_emas"] = vpu_LUT_para_lut_emas
    pre_process_dict["vpu_LUT_para_lut_ret1n"] = vpu_LUT_para_lut_ret1n
    pre_process_dict["vpu_LUT_para_lut_rawl"] = vpu_LUT_para_lut_rawl
    pre_process_dict["vpu_LUT_para_lut_rawlm"] = vpu_LUT_para_lut_rawlm
    pre_process_dict["vpu_LUT_para_lut_wabl"] = vpu_LUT_para_lut_wabl
    pre_process_dict["vpu_LUT_para_lut_wablm"] = vpu_LUT_para_lut_wablm
    pre_process_dict["vpu_UNIT_para_i_vpu_unit_seed"] = vpu_UNIT_para_i_vpu_unit_seed
    pre_process_dict["vpu_INTERFACE_para_i_vpu_interface_seed"] = vpu_INTERFACE_para_i_vpu_interface_seed
    pre_process_dict["vpu_IN_para_i_vpu_in_fifo_group_seed"] = vpu_IN_para_i_vpu_in_fifo_group_seed
    pre_process_dict["vpu_DATA_para_i_target_data_amount"] = vpu_DATA_para_i_target_data_amount
    pre_process_dict["vpu_BIST_para_o_ans_data"] = vpu_BIST_para_o_ans_data
    pre_process_dict["vpu_BIST_para_o_ans_data_sc"] = vpu_BIST_para_o_ans_data_sc
    pre_process_dict["vpu_BIST_para_o_ans_data_wo_q"] = vpu_BIST_para_o_ans_data_wo_q
    pre_process_dict["vpu_BIST_para_o_ans_done_flag"] = vpu_BIST_para_o_ans_done_flag
    pre_process_dict["vpu_BIST_para_o_ans_col_done"] = vpu_BIST_para_o_ans_col_done
    pre_process_dict["vpu_BIST_para_all_ans_data_out_psum"] = vpu_BIST_para_all_ans_data_out_psum
    pre_process_dict["vpu_BIST_para_all_ans_data_out_0"] = vpu_BIST_para_all_ans_data_out_0
    pre_process_dict["vpu_BIST_para_all_ans_data_out_1"] = vpu_BIST_para_all_ans_data_out_1
    pre_process_dict["vpu_BIST_para_all_ans_data_out_2"] = vpu_BIST_para_all_ans_data_out_2
    pre_process_dict["vpu_BIST_para_all_ans_data_out_3"] = vpu_BIST_para_all_ans_data_out_3
    pre_process_dict["vpu_BIST_para_o_b0_ans_data"] = vpu_BIST_para_o_b0_ans_data
    pre_process_dict["vpu_BIST_para_o_b0_ans_ctrl"] = vpu_BIST_para_o_b0_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b0_ans_addr"] = vpu_BIST_para_o_b0_ans_addr
    pre_process_dict["vpu_BIST_para_o_b1_ans_data"] = vpu_BIST_para_o_b1_ans_data
    pre_process_dict["vpu_BIST_para_o_b1_ans_ctrl"] = vpu_BIST_para_o_b1_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b1_ans_addr"] = vpu_BIST_para_o_b1_ans_addr
    pre_process_dict["vpu_BIST_para_o_b2_ans_data"] = vpu_BIST_para_o_b2_ans_data
    pre_process_dict["vpu_BIST_para_o_b2_ans_ctrl"] = vpu_BIST_para_o_b2_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b2_ans_addr"] = vpu_BIST_para_o_b2_ans_addr
    pre_process_dict["vpu_BIST_para_o_b3_ans_data"] = vpu_BIST_para_o_b3_ans_data
    pre_process_dict["vpu_BIST_para_o_b3_ans_ctrl"] = vpu_BIST_para_o_b3_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b3_ans_addr"] = vpu_BIST_para_o_b3_ans_addr
    pre_process_dict["vpu_BIST_para_o_b4_ans_data"] = vpu_BIST_para_o_b4_ans_data
    pre_process_dict["vpu_BIST_para_o_b4_ans_ctrl"] = vpu_BIST_para_o_b4_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b4_ans_addr"] = vpu_BIST_para_o_b4_ans_addr
    pre_process_dict["vpu_BIST_para_o_b5_ans_data"] = vpu_BIST_para_o_b5_ans_data
    pre_process_dict["vpu_BIST_para_o_b5_ans_ctrl"] = vpu_BIST_para_o_b5_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b5_ans_addr"] = vpu_BIST_para_o_b5_ans_addr
    pre_process_dict["vpu_BIST_para_o_b6_ans_data"] = vpu_BIST_para_o_b6_ans_data
    pre_process_dict["vpu_BIST_para_o_b6_ans_ctrl"] = vpu_BIST_para_o_b6_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b6_ans_addr"] = vpu_BIST_para_o_b6_ans_addr
    pre_process_dict["vpu_BIST_para_o_b7_ans_data"] = vpu_BIST_para_o_b7_ans_data
    pre_process_dict["vpu_BIST_para_o_b7_ans_ctrl"] = vpu_BIST_para_o_b7_ans_ctrl
    pre_process_dict["vpu_BIST_para_o_b7_ans_addr"] = vpu_BIST_para_o_b7_ans_addr
    pre_process_dict["vpu_0_para_clus_0_scw_ad_0"] = vpu_0_para_clus_0_scw_ad_0
    pre_process_dict["vpu_0_para_clus_0_scr_ad_0"] = vpu_0_para_clus_0_scr_ad_0
    pre_process_dict["vpu_0_para_clus_0_ad_0"] = vpu_0_para_clus_0_ad_0
    pre_process_dict["vpu_0_para_clus_0_scw_ad_1"] = vpu_0_para_clus_0_scw_ad_1
    pre_process_dict["vpu_0_para_clus_0_scr_ad_1"] = vpu_0_para_clus_0_scr_ad_1
    pre_process_dict["vpu_0_para_clus_0_ad_1"] = vpu_0_para_clus_0_ad_1
    pre_process_dict["vpu_0_para_clus_0_scw_ad_2"] = vpu_0_para_clus_0_scw_ad_2
    pre_process_dict["vpu_0_para_clus_0_scr_ad_2"] = vpu_0_para_clus_0_scr_ad_2
    pre_process_dict["vpu_0_para_clus_0_ad_2"] = vpu_0_para_clus_0_ad_2
    pre_process_dict["vpu_0_para_clus_0_scw_ad_3"] = vpu_0_para_clus_0_scw_ad_3
    pre_process_dict["vpu_0_para_clus_0_scr_ad_3"] = vpu_0_para_clus_0_scr_ad_3
    pre_process_dict["vpu_0_para_clus_0_ad_3"] = vpu_0_para_clus_0_ad_3
    pre_process_dict["vpu_0_para_clus_0_block_ad_jump_0"] = vpu_0_para_clus_0_block_ad_jump_0
    pre_process_dict["vpu_0_para_clus_0_block_ad_mode_enable"] = vpu_0_para_clus_0_block_ad_mode_enable
    pre_process_dict["vpu_0_para_clus_0_block_ad_jump_condit1"] = vpu_0_para_clus_0_block_ad_jump_condit1
    pre_process_dict["vpu_0_para_clus_0_block_ad_jump_condit0"] = vpu_0_para_clus_0_block_ad_jump_condit0
    pre_process_dict["vpu_0_para_clus_0_block_ad_jump_2"] = vpu_0_para_clus_0_block_ad_jump_2
    pre_process_dict["vpu_0_para_clus_0_block_ad_jump_1"] = vpu_0_para_clus_0_block_ad_jump_1
    pre_process_dict["vpu_0_para_clus_0_block_scr_ad_jump_0"] = vpu_0_para_clus_0_block_scr_ad_jump_0
    pre_process_dict["vpu_0_para_clus_0_block_scr_ad_jump_condit1"] = vpu_0_para_clus_0_block_scr_ad_jump_condit1
    pre_process_dict["vpu_0_para_clus_0_block_scr_ad_jump_condit0"] = vpu_0_para_clus_0_block_scr_ad_jump_condit0
    pre_process_dict["vpu_0_para_clus_0_block_scr_ad_jump_2"] = vpu_0_para_clus_0_block_scr_ad_jump_2
    pre_process_dict["vpu_0_para_clus_0_block_scr_ad_jump_1"] = vpu_0_para_clus_0_block_scr_ad_jump_1
    pre_process_dict["vpu_0_para_clus_0_block_scw_ad_jump_0"] = vpu_0_para_clus_0_block_scw_ad_jump_0
    pre_process_dict["vpu_0_para_clus_0_block_scw_ad_jump_condit1"] = vpu_0_para_clus_0_block_scw_ad_jump_condit1
    pre_process_dict["vpu_0_para_clus_0_block_scw_ad_jump_condit0"] = vpu_0_para_clus_0_block_scw_ad_jump_condit0
    pre_process_dict["vpu_0_para_clus_0_block_scw_ad_jump_2"] = vpu_0_para_clus_0_block_scw_ad_jump_2
    pre_process_dict["vpu_0_para_clus_0_block_scw_ad_jump_1"] = vpu_0_para_clus_0_block_scw_ad_jump_1
    pre_process_dict["vpu_0_para_clus_0_line_buffer_w_max"] = vpu_0_para_clus_0_line_buffer_w_max
    pre_process_dict["vpu_0_para_clus_0_line_buffer_h_max"] = vpu_0_para_clus_0_line_buffer_h_max
    pre_process_dict["vpu_0_para_clus_0_kernal_h"] = vpu_0_para_clus_0_kernal_h
    pre_process_dict["vpu_0_para_clus_0_kernal_h_stride"] = vpu_0_para_clus_0_kernal_h_stride
    pre_process_dict["vpu_0_para_clus_0_output_l1_step"] = vpu_0_para_clus_0_output_l1_step
    pre_process_dict["vpu_0_para_clus_0_output_l1_condition"] = vpu_0_para_clus_0_output_l1_condition
    pre_process_dict["vpu_0_para_clus_0_output_l2_step"] = vpu_0_para_clus_0_output_l2_step
    pre_process_dict["vpu_0_para_clus_0_output_l2_condition"] = vpu_0_para_clus_0_output_l2_condition
    pre_process_dict["vpu_0_para_clus_0_output_l3_step"] = vpu_0_para_clus_0_output_l3_step
    pre_process_dict["vpu_0_para_clus_0_output_l3_condition"] = vpu_0_para_clus_0_output_l3_condition
    pre_process_dict["vpu_0_para_clus_0_output_l1_addr_step"] = vpu_0_para_clus_0_output_l1_addr_step
    pre_process_dict["vpu_0_para_clus_0_output_l2_addr_step"] = vpu_0_para_clus_0_output_l2_addr_step
    pre_process_dict["vpu_0_para_clus_0_output_l3_addr_step"] = vpu_0_para_clus_0_output_l3_addr_step
    pre_process_dict["vpu_0_para_clus_0_scw_l1_step"] = vpu_0_para_clus_0_scw_l1_step
    pre_process_dict["vpu_0_para_clus_0_scw_l1_condition"] = vpu_0_para_clus_0_scw_l1_condition
    pre_process_dict["vpu_0_para_clus_0_scw_l2_step"] = vpu_0_para_clus_0_scw_l2_step
    pre_process_dict["vpu_0_para_clus_0_scw_l2_condition"] = vpu_0_para_clus_0_scw_l2_condition
    pre_process_dict["vpu_0_para_clus_0_scw_l3_step"] = vpu_0_para_clus_0_scw_l3_step
    pre_process_dict["vpu_0_para_clus_0_scw_l3_condition"] = vpu_0_para_clus_0_scw_l3_condition
    pre_process_dict["vpu_0_para_clus_0_scw_l1_addr_step"] = vpu_0_para_clus_0_scw_l1_addr_step
    pre_process_dict["vpu_0_para_clus_0_scw_l2_addr_step"] = vpu_0_para_clus_0_scw_l2_addr_step
    pre_process_dict["vpu_0_para_clus_0_scw_l3_addr_step"] = vpu_0_para_clus_0_scw_l3_addr_step
    pre_process_dict["vpu_0_para_clus_0_scr_l1_step"] = vpu_0_para_clus_0_scr_l1_step
    pre_process_dict["vpu_0_para_clus_0_scr_l1_condition"] = vpu_0_para_clus_0_scr_l1_condition
    pre_process_dict["vpu_0_para_clus_0_scr_l2_step"] = vpu_0_para_clus_0_scr_l2_step
    pre_process_dict["vpu_0_para_clus_0_scr_l2_condition"] = vpu_0_para_clus_0_scr_l2_condition
    pre_process_dict["vpu_0_para_clus_0_scr_l3_step"] = vpu_0_para_clus_0_scr_l3_step
    pre_process_dict["vpu_0_para_clus_0_scr_l3_condition"] = vpu_0_para_clus_0_scr_l3_condition
    pre_process_dict["vpu_0_para_clus_0_scr_l1_addr_step"] = vpu_0_para_clus_0_scr_l1_addr_step
    pre_process_dict["vpu_0_para_clus_0_scr_l2_addr_step"] = vpu_0_para_clus_0_scr_l2_addr_step
    pre_process_dict["vpu_0_para_clus_0_scr_l3_addr_step"] = vpu_0_para_clus_0_scr_l3_addr_step
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_width_ratio"] = vpu_0_para_i_clus_0_resize_param_width_ratio
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_height_ratio"] = vpu_0_para_i_clus_0_resize_param_height_ratio
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_input_width"] = vpu_0_para_i_clus_0_resize_param_input_width
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_input_height"] = vpu_0_para_i_clus_0_resize_param_input_height
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_output_width"] = vpu_0_para_i_clus_0_resize_param_output_width
    pre_process_dict["vpu_0_para_i_clus_0_resize_param_output_height"] = vpu_0_para_i_clus_0_resize_param_output_height
    pre_process_dict["vpu_0_para_i_clus_0_pooling_param_input_width"] = vpu_0_para_i_clus_0_pooling_param_input_width
    pre_process_dict["vpu_0_para_i_clus_0_pooling_param_input_height"] = vpu_0_para_i_clus_0_pooling_param_input_height
    pre_process_dict["vpu_0_para_i_clus_0_pooling_param_output_width"] = vpu_0_para_i_clus_0_pooling_param_output_width
    pre_process_dict[
        "vpu_0_para_i_clus_0_pooling_param_output_height"] = vpu_0_para_i_clus_0_pooling_param_output_height
    pre_process_dict["vpu_0_para_i_clus_0_pooling_padding_mode"] = vpu_0_para_i_clus_0_pooling_padding_mode
    pre_process_dict["vpu_0_para_i_clus_0_pooling_padding_width"] = vpu_0_para_i_clus_0_pooling_padding_width
    pre_process_dict["vpu_0_para_i_clus_0_pooling_padding_height"] = vpu_0_para_i_clus_0_pooling_padding_height
    pre_process_dict["vpu_1_para_clus_1_scw_ad_0"] = vpu_1_para_clus_1_scw_ad_0
    pre_process_dict["vpu_1_para_clus_1_scr_ad_0"] = vpu_1_para_clus_1_scr_ad_0
    pre_process_dict["vpu_1_para_clus_1_ad_0"] = vpu_1_para_clus_1_ad_0
    pre_process_dict["vpu_1_para_clus_1_scw_ad_1"] = vpu_1_para_clus_1_scw_ad_1
    pre_process_dict["vpu_1_para_clus_1_scr_ad_1"] = vpu_1_para_clus_1_scr_ad_1
    pre_process_dict["vpu_1_para_clus_1_ad_1"] = vpu_1_para_clus_1_ad_1
    pre_process_dict["vpu_1_para_clus_1_scw_ad_2"] = vpu_1_para_clus_1_scw_ad_2
    pre_process_dict["vpu_1_para_clus_1_scr_ad_2"] = vpu_1_para_clus_1_scr_ad_2
    pre_process_dict["vpu_1_para_clus_1_ad_2"] = vpu_1_para_clus_1_ad_2
    pre_process_dict["vpu_1_para_clus_1_scw_ad_3"] = vpu_1_para_clus_1_scw_ad_3
    pre_process_dict["vpu_1_para_clus_1_scr_ad_3"] = vpu_1_para_clus_1_scr_ad_3
    pre_process_dict["vpu_1_para_clus_1_ad_3"] = vpu_1_para_clus_1_ad_3
    pre_process_dict["vpu_1_para_clus_1_block_ad_jump_0"] = vpu_1_para_clus_1_block_ad_jump_0
    pre_process_dict["vpu_1_para_clus_1_block_ad_mode_enable"] = vpu_1_para_clus_1_block_ad_mode_enable
    pre_process_dict["vpu_1_para_clus_1_block_ad_jump_condit1"] = vpu_1_para_clus_1_block_ad_jump_condit1
    pre_process_dict["vpu_1_para_clus_1_block_ad_jump_condit0"] = vpu_1_para_clus_1_block_ad_jump_condit0
    pre_process_dict["vpu_1_para_clus_1_block_ad_jump_2"] = vpu_1_para_clus_1_block_ad_jump_2
    pre_process_dict["vpu_1_para_clus_1_block_ad_jump_1"] = vpu_1_para_clus_1_block_ad_jump_1
    pre_process_dict["vpu_1_para_clus_1_block_scr_ad_jump_0"] = vpu_1_para_clus_1_block_scr_ad_jump_0
    pre_process_dict["vpu_1_para_clus_1_block_scr_ad_jump_condit1"] = vpu_1_para_clus_1_block_scr_ad_jump_condit1
    pre_process_dict["vpu_1_para_clus_1_block_scr_ad_jump_condit0"] = vpu_1_para_clus_1_block_scr_ad_jump_condit0
    pre_process_dict["vpu_1_para_clus_1_block_scr_ad_jump_2"] = vpu_1_para_clus_1_block_scr_ad_jump_2
    pre_process_dict["vpu_1_para_clus_1_block_scr_ad_jump_1"] = vpu_1_para_clus_1_block_scr_ad_jump_1
    pre_process_dict["vpu_1_para_clus_1_block_scw_ad_jump_0"] = vpu_1_para_clus_1_block_scw_ad_jump_0
    pre_process_dict["vpu_1_para_clus_1_block_scw_ad_jump_condit1"] = vpu_1_para_clus_1_block_scw_ad_jump_condit1
    pre_process_dict["vpu_1_para_clus_1_block_scw_ad_jump_condit0"] = vpu_1_para_clus_1_block_scw_ad_jump_condit0
    pre_process_dict["vpu_1_para_clus_1_block_scw_ad_jump_2"] = vpu_1_para_clus_1_block_scw_ad_jump_2
    pre_process_dict["vpu_1_para_clus_1_block_scw_ad_jump_1"] = vpu_1_para_clus_1_block_scw_ad_jump_1
    pre_process_dict["vpu_1_para_clus_1_line_buffer_w_max"] = vpu_1_para_clus_1_line_buffer_w_max
    pre_process_dict["vpu_1_para_clus_1_line_buffer_h_max"] = vpu_1_para_clus_1_line_buffer_h_max
    pre_process_dict["vpu_1_para_clus_1_kernal_h"] = vpu_1_para_clus_1_kernal_h
    pre_process_dict["vpu_1_para_clus_1_kernal_h_stride"] = vpu_1_para_clus_1_kernal_h_stride
    pre_process_dict["vpu_1_para_clus_1_output_l1_step"] = vpu_1_para_clus_1_output_l1_step
    pre_process_dict["vpu_1_para_clus_1_output_l1_condition"] = vpu_1_para_clus_1_output_l1_condition
    pre_process_dict["vpu_1_para_clus_1_output_l2_step"] = vpu_1_para_clus_1_output_l2_step
    pre_process_dict["vpu_1_para_clus_1_output_l2_condition"] = vpu_1_para_clus_1_output_l2_condition
    pre_process_dict["vpu_1_para_clus_1_output_l3_step"] = vpu_1_para_clus_1_output_l3_step
    pre_process_dict["vpu_1_para_clus_1_output_l3_condition"] = vpu_1_para_clus_1_output_l3_condition
    pre_process_dict["vpu_1_para_clus_1_output_l1_addr_step"] = vpu_1_para_clus_1_output_l1_addr_step
    pre_process_dict["vpu_1_para_clus_1_output_l2_addr_step"] = vpu_1_para_clus_1_output_l2_addr_step
    pre_process_dict["vpu_1_para_clus_1_output_l3_addr_step"] = vpu_1_para_clus_1_output_l3_addr_step
    pre_process_dict["vpu_1_para_clus_1_scw_l1_step"] = vpu_1_para_clus_1_scw_l1_step
    pre_process_dict["vpu_1_para_clus_1_scw_l1_condition"] = vpu_1_para_clus_1_scw_l1_condition
    pre_process_dict["vpu_1_para_clus_1_scw_l2_step"] = vpu_1_para_clus_1_scw_l2_step
    pre_process_dict["vpu_1_para_clus_1_scw_l2_condition"] = vpu_1_para_clus_1_scw_l2_condition
    pre_process_dict["vpu_1_para_clus_1_scw_l3_step"] = vpu_1_para_clus_1_scw_l3_step
    pre_process_dict["vpu_1_para_clus_1_scw_l3_condition"] = vpu_1_para_clus_1_scw_l3_condition
    pre_process_dict["vpu_1_para_clus_1_scw_l1_addr_step"] = vpu_1_para_clus_1_scw_l1_addr_step
    pre_process_dict["vpu_1_para_clus_1_scw_l2_addr_step"] = vpu_1_para_clus_1_scw_l2_addr_step
    pre_process_dict["vpu_1_para_clus_1_scw_l3_addr_step"] = vpu_1_para_clus_1_scw_l3_addr_step
    pre_process_dict["vpu_1_para_clus_1_scr_l1_step"] = vpu_1_para_clus_1_scr_l1_step
    pre_process_dict["vpu_1_para_clus_1_scr_l1_condition"] = vpu_1_para_clus_1_scr_l1_condition
    pre_process_dict["vpu_1_para_clus_1_scr_l2_step"] = vpu_1_para_clus_1_scr_l2_step
    pre_process_dict["vpu_1_para_clus_1_scr_l2_condition"] = vpu_1_para_clus_1_scr_l2_condition
    pre_process_dict["vpu_1_para_clus_1_scr_l3_step"] = vpu_1_para_clus_1_scr_l3_step
    pre_process_dict["vpu_1_para_clus_1_scr_l3_condition"] = vpu_1_para_clus_1_scr_l3_condition
    pre_process_dict["vpu_1_para_clus_1_scr_l1_addr_step"] = vpu_1_para_clus_1_scr_l1_addr_step
    pre_process_dict["vpu_1_para_clus_1_scr_l2_addr_step"] = vpu_1_para_clus_1_scr_l2_addr_step
    pre_process_dict["vpu_1_para_clus_1_scr_l3_addr_step"] = vpu_1_para_clus_1_scr_l3_addr_step
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_width_ratio"] = vpu_1_para_i_clus_1_resize_param_width_ratio
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_height_ratio"] = vpu_1_para_i_clus_1_resize_param_height_ratio
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_input_width"] = vpu_1_para_i_clus_1_resize_param_input_width
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_input_height"] = vpu_1_para_i_clus_1_resize_param_input_height
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_output_width"] = vpu_1_para_i_clus_1_resize_param_output_width
    pre_process_dict["vpu_1_para_i_clus_1_resize_param_output_height"] = vpu_1_para_i_clus_1_resize_param_output_height
    pre_process_dict["vpu_1_para_i_clus_1_pooling_param_input_width"] = vpu_1_para_i_clus_1_pooling_param_input_width
    pre_process_dict["vpu_1_para_i_clus_1_pooling_param_input_height"] = vpu_1_para_i_clus_1_pooling_param_input_height
    pre_process_dict["vpu_1_para_i_clus_1_pooling_param_output_width"] = vpu_1_para_i_clus_1_pooling_param_output_width
    pre_process_dict[
        "vpu_1_para_i_clus_1_pooling_param_output_height"] = vpu_1_para_i_clus_1_pooling_param_output_height
    pre_process_dict["vpu_1_para_i_clus_1_pooling_padding_mode"] = vpu_1_para_i_clus_1_pooling_padding_mode
    pre_process_dict["vpu_1_para_i_clus_1_pooling_padding_width"] = vpu_1_para_i_clus_1_pooling_padding_width
    pre_process_dict["vpu_1_para_i_clus_1_pooling_padding_height"] = vpu_1_para_i_clus_1_pooling_padding_height
    pre_process_dict["vpu_2_para_clus_2_scw_ad_0"] = vpu_2_para_clus_2_scw_ad_0
    pre_process_dict["vpu_2_para_clus_2_scr_ad_0"] = vpu_2_para_clus_2_scr_ad_0
    pre_process_dict["vpu_2_para_clus_2_ad_0"] = vpu_2_para_clus_2_ad_0
    pre_process_dict["vpu_2_para_clus_2_scw_ad_1"] = vpu_2_para_clus_2_scw_ad_1
    pre_process_dict["vpu_2_para_clus_2_scr_ad_1"] = vpu_2_para_clus_2_scr_ad_1
    pre_process_dict["vpu_2_para_clus_2_ad_1"] = vpu_2_para_clus_2_ad_1
    pre_process_dict["vpu_2_para_clus_2_scw_ad_2"] = vpu_2_para_clus_2_scw_ad_2
    pre_process_dict["vpu_2_para_clus_2_scr_ad_2"] = vpu_2_para_clus_2_scr_ad_2
    pre_process_dict["vpu_2_para_clus_2_ad_2"] = vpu_2_para_clus_2_ad_2
    pre_process_dict["vpu_2_para_clus_2_scw_ad_3"] = vpu_2_para_clus_2_scw_ad_3
    pre_process_dict["vpu_2_para_clus_2_scr_ad_3"] = vpu_2_para_clus_2_scr_ad_3
    pre_process_dict["vpu_2_para_clus_2_ad_3"] = vpu_2_para_clus_2_ad_3
    pre_process_dict["vpu_2_para_clus_2_block_ad_jump_0"] = vpu_2_para_clus_2_block_ad_jump_0
    pre_process_dict["vpu_2_para_clus_2_block_ad_mode_enable"] = vpu_2_para_clus_2_block_ad_mode_enable
    pre_process_dict["vpu_2_para_clus_2_block_ad_jump_condit1"] = vpu_2_para_clus_2_block_ad_jump_condit1
    pre_process_dict["vpu_2_para_clus_2_block_ad_jump_condit0"] = vpu_2_para_clus_2_block_ad_jump_condit0
    pre_process_dict["vpu_2_para_clus_2_block_ad_jump_2"] = vpu_2_para_clus_2_block_ad_jump_2
    pre_process_dict["vpu_2_para_clus_2_block_ad_jump_1"] = vpu_2_para_clus_2_block_ad_jump_1
    pre_process_dict["vpu_2_para_clus_2_block_scr_ad_jump_0"] = vpu_2_para_clus_2_block_scr_ad_jump_0
    pre_process_dict["vpu_2_para_clus_2_block_scr_ad_jump_condit1"] = vpu_2_para_clus_2_block_scr_ad_jump_condit1
    pre_process_dict["vpu_2_para_clus_2_block_scr_ad_jump_condit0"] = vpu_2_para_clus_2_block_scr_ad_jump_condit0
    pre_process_dict["vpu_2_para_clus_2_block_scr_ad_jump_2"] = vpu_2_para_clus_2_block_scr_ad_jump_2
    pre_process_dict["vpu_2_para_clus_2_block_scr_ad_jump_1"] = vpu_2_para_clus_2_block_scr_ad_jump_1
    pre_process_dict["vpu_2_para_clus_2_block_scw_ad_jump_0"] = vpu_2_para_clus_2_block_scw_ad_jump_0
    pre_process_dict["vpu_2_para_clus_2_block_scw_ad_jump_condit1"] = vpu_2_para_clus_2_block_scw_ad_jump_condit1
    pre_process_dict["vpu_2_para_clus_2_block_scw_ad_jump_condit0"] = vpu_2_para_clus_2_block_scw_ad_jump_condit0
    pre_process_dict["vpu_2_para_clus_2_block_scw_ad_jump_2"] = vpu_2_para_clus_2_block_scw_ad_jump_2
    pre_process_dict["vpu_2_para_clus_2_block_scw_ad_jump_1"] = vpu_2_para_clus_2_block_scw_ad_jump_1
    pre_process_dict["vpu_2_para_clus_2_line_buffer_w_max"] = vpu_2_para_clus_2_line_buffer_w_max
    pre_process_dict["vpu_2_para_clus_2_line_buffer_h_max"] = vpu_2_para_clus_2_line_buffer_h_max
    pre_process_dict["vpu_2_para_clus_2_kernal_h"] = vpu_2_para_clus_2_kernal_h
    pre_process_dict["vpu_2_para_clus_2_kernal_h_stride"] = vpu_2_para_clus_2_kernal_h_stride
    pre_process_dict["vpu_2_para_clus_2_output_l1_step"] = vpu_2_para_clus_2_output_l1_step
    pre_process_dict["vpu_2_para_clus_2_output_l1_condition"] = vpu_2_para_clus_2_output_l1_condition
    pre_process_dict["vpu_2_para_clus_2_output_l2_step"] = vpu_2_para_clus_2_output_l2_step
    pre_process_dict["vpu_2_para_clus_2_output_l2_condition"] = vpu_2_para_clus_2_output_l2_condition
    pre_process_dict["vpu_2_para_clus_2_output_l3_step"] = vpu_2_para_clus_2_output_l3_step
    pre_process_dict["vpu_2_para_clus_2_output_l3_condition"] = vpu_2_para_clus_2_output_l3_condition
    pre_process_dict["vpu_2_para_clus_2_output_l1_addr_step"] = vpu_2_para_clus_2_output_l1_addr_step
    pre_process_dict["vpu_2_para_clus_2_output_l2_addr_step"] = vpu_2_para_clus_2_output_l2_addr_step
    pre_process_dict["vpu_2_para_clus_2_output_l3_addr_step"] = vpu_2_para_clus_2_output_l3_addr_step
    pre_process_dict["vpu_2_para_clus_2_scw_l1_step"] = vpu_2_para_clus_2_scw_l1_step
    pre_process_dict["vpu_2_para_clus_2_scw_l1_condition"] = vpu_2_para_clus_2_scw_l1_condition
    pre_process_dict["vpu_2_para_clus_2_scw_l2_step"] = vpu_2_para_clus_2_scw_l2_step
    pre_process_dict["vpu_2_para_clus_2_scw_l2_condition"] = vpu_2_para_clus_2_scw_l2_condition
    pre_process_dict["vpu_2_para_clus_2_scw_l3_step"] = vpu_2_para_clus_2_scw_l3_step
    pre_process_dict["vpu_2_para_clus_2_scw_l3_condition"] = vpu_2_para_clus_2_scw_l3_condition
    pre_process_dict["vpu_2_para_clus_2_scw_l1_addr_step"] = vpu_2_para_clus_2_scw_l1_addr_step
    pre_process_dict["vpu_2_para_clus_2_scw_l2_addr_step"] = vpu_2_para_clus_2_scw_l2_addr_step
    pre_process_dict["vpu_2_para_clus_2_scw_l3_addr_step"] = vpu_2_para_clus_2_scw_l3_addr_step
    pre_process_dict["vpu_2_para_clus_2_scr_l1_step"] = vpu_2_para_clus_2_scr_l1_step
    pre_process_dict["vpu_2_para_clus_2_scr_l1_condition"] = vpu_2_para_clus_2_scr_l1_condition
    pre_process_dict["vpu_2_para_clus_2_scr_l2_step"] = vpu_2_para_clus_2_scr_l2_step
    pre_process_dict["vpu_2_para_clus_2_scr_l2_condition"] = vpu_2_para_clus_2_scr_l2_condition
    pre_process_dict["vpu_2_para_clus_2_scr_l3_step"] = vpu_2_para_clus_2_scr_l3_step
    pre_process_dict["vpu_2_para_clus_2_scr_l3_condition"] = vpu_2_para_clus_2_scr_l3_condition
    pre_process_dict["vpu_2_para_clus_2_scr_l1_addr_step"] = vpu_2_para_clus_2_scr_l1_addr_step
    pre_process_dict["vpu_2_para_clus_2_scr_l2_addr_step"] = vpu_2_para_clus_2_scr_l2_addr_step
    pre_process_dict["vpu_2_para_clus_2_scr_l3_addr_step"] = vpu_2_para_clus_2_scr_l3_addr_step
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_width_ratio"] = vpu_2_para_i_clus_2_resize_param_width_ratio
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_height_ratio"] = vpu_2_para_i_clus_2_resize_param_height_ratio
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_input_width"] = vpu_2_para_i_clus_2_resize_param_input_width
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_input_height"] = vpu_2_para_i_clus_2_resize_param_input_height
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_output_width"] = vpu_2_para_i_clus_2_resize_param_output_width
    pre_process_dict["vpu_2_para_i_clus_2_resize_param_output_height"] = vpu_2_para_i_clus_2_resize_param_output_height
    pre_process_dict["vpu_2_para_i_clus_2_pooling_param_input_width"] = vpu_2_para_i_clus_2_pooling_param_input_width
    pre_process_dict["vpu_2_para_i_clus_2_pooling_param_input_height"] = vpu_2_para_i_clus_2_pooling_param_input_height
    pre_process_dict["vpu_2_para_i_clus_2_pooling_param_output_width"] = vpu_2_para_i_clus_2_pooling_param_output_width
    pre_process_dict[
        "vpu_2_para_i_clus_2_pooling_param_output_height"] = vpu_2_para_i_clus_2_pooling_param_output_height
    pre_process_dict["vpu_2_para_i_clus_2_pooling_padding_mode"] = vpu_2_para_i_clus_2_pooling_padding_mode
    pre_process_dict["vpu_2_para_i_clus_2_pooling_padding_width"] = vpu_2_para_i_clus_2_pooling_padding_width
    pre_process_dict["vpu_2_para_i_clus_2_pooling_padding_height"] = vpu_2_para_i_clus_2_pooling_padding_height
    pre_process_dict["vpu_3_para_clus_3_scw_ad_0"] = vpu_3_para_clus_3_scw_ad_0
    pre_process_dict["vpu_3_para_clus_3_scr_ad_0"] = vpu_3_para_clus_3_scr_ad_0
    pre_process_dict["vpu_3_para_clus_3_ad_0"] = vpu_3_para_clus_3_ad_0
    pre_process_dict["vpu_3_para_clus_3_scw_ad_1"] = vpu_3_para_clus_3_scw_ad_1
    pre_process_dict["vpu_3_para_clus_3_scr_ad_1"] = vpu_3_para_clus_3_scr_ad_1
    pre_process_dict["vpu_3_para_clus_3_ad_1"] = vpu_3_para_clus_3_ad_1
    pre_process_dict["vpu_3_para_clus_3_scw_ad_2"] = vpu_3_para_clus_3_scw_ad_2
    pre_process_dict["vpu_3_para_clus_3_scr_ad_2"] = vpu_3_para_clus_3_scr_ad_2
    pre_process_dict["vpu_3_para_clus_3_ad_2"] = vpu_3_para_clus_3_ad_2
    pre_process_dict["vpu_3_para_clus_3_scw_ad_3"] = vpu_3_para_clus_3_scw_ad_3
    pre_process_dict["vpu_3_para_clus_3_scr_ad_3"] = vpu_3_para_clus_3_scr_ad_3
    pre_process_dict["vpu_3_para_clus_3_ad_3"] = vpu_3_para_clus_3_ad_3
    pre_process_dict["vpu_3_para_clus_3_block_ad_jump_0"] = vpu_3_para_clus_3_block_ad_jump_0
    pre_process_dict["vpu_3_para_clus_3_block_ad_mode_enable"] = vpu_3_para_clus_3_block_ad_mode_enable
    pre_process_dict["vpu_3_para_clus_3_block_ad_jump_condit1"] = vpu_3_para_clus_3_block_ad_jump_condit1
    pre_process_dict["vpu_3_para_clus_3_block_ad_jump_condit0"] = vpu_3_para_clus_3_block_ad_jump_condit0
    pre_process_dict["vpu_3_para_clus_3_block_ad_jump_2"] = vpu_3_para_clus_3_block_ad_jump_2
    pre_process_dict["vpu_3_para_clus_3_block_ad_jump_1"] = vpu_3_para_clus_3_block_ad_jump_1
    pre_process_dict["vpu_3_para_clus_3_block_scr_ad_jump_0"] = vpu_3_para_clus_3_block_scr_ad_jump_0
    pre_process_dict["vpu_3_para_clus_3_block_scr_ad_jump_condit1"] = vpu_3_para_clus_3_block_scr_ad_jump_condit1
    pre_process_dict["vpu_3_para_clus_3_block_scr_ad_jump_condit0"] = vpu_3_para_clus_3_block_scr_ad_jump_condit0
    pre_process_dict["vpu_3_para_clus_3_block_scr_ad_jump_2"] = vpu_3_para_clus_3_block_scr_ad_jump_2
    pre_process_dict["vpu_3_para_clus_3_block_scr_ad_jump_1"] = vpu_3_para_clus_3_block_scr_ad_jump_1
    pre_process_dict["vpu_3_para_clus_3_block_scw_ad_jump_0"] = vpu_3_para_clus_3_block_scw_ad_jump_0
    pre_process_dict["vpu_3_para_clus_3_block_scw_ad_jump_condit1"] = vpu_3_para_clus_3_block_scw_ad_jump_condit1
    pre_process_dict["vpu_3_para_clus_3_block_scw_ad_jump_condit0"] = vpu_3_para_clus_3_block_scw_ad_jump_condit0
    pre_process_dict["vpu_3_para_clus_3_block_scw_ad_jump_2"] = vpu_3_para_clus_3_block_scw_ad_jump_2
    pre_process_dict["vpu_3_para_clus_3_block_scw_ad_jump_1"] = vpu_3_para_clus_3_block_scw_ad_jump_1
    pre_process_dict["vpu_3_para_clus_3_line_buffer_w_max"] = vpu_3_para_clus_3_line_buffer_w_max
    pre_process_dict["vpu_3_para_clus_3_line_buffer_h_max"] = vpu_3_para_clus_3_line_buffer_h_max
    pre_process_dict["vpu_3_para_clus_3_kernal_h"] = vpu_3_para_clus_3_kernal_h
    pre_process_dict["vpu_3_para_clus_3_kernal_h_stride"] = vpu_3_para_clus_3_kernal_h_stride
    pre_process_dict["vpu_3_para_clus_3_output_l1_step"] = vpu_3_para_clus_3_output_l1_step
    pre_process_dict["vpu_3_para_clus_3_output_l1_condition"] = vpu_3_para_clus_3_output_l1_condition
    pre_process_dict["vpu_3_para_clus_3_output_l2_step"] = vpu_3_para_clus_3_output_l2_step
    pre_process_dict["vpu_3_para_clus_3_output_l2_condition"] = vpu_3_para_clus_3_output_l2_condition
    pre_process_dict["vpu_3_para_clus_3_output_l3_step"] = vpu_3_para_clus_3_output_l3_step
    pre_process_dict["vpu_3_para_clus_3_output_l3_condition"] = vpu_3_para_clus_3_output_l3_condition
    pre_process_dict["vpu_3_para_clus_3_output_l1_addr_step"] = vpu_3_para_clus_3_output_l1_addr_step
    pre_process_dict["vpu_3_para_clus_3_output_l2_addr_step"] = vpu_3_para_clus_3_output_l2_addr_step
    pre_process_dict["vpu_3_para_clus_3_output_l3_addr_step"] = vpu_3_para_clus_3_output_l3_addr_step
    pre_process_dict["vpu_3_para_clus_3_scw_l1_step"] = vpu_3_para_clus_3_scw_l1_step
    pre_process_dict["vpu_3_para_clus_3_scw_l1_condition"] = vpu_3_para_clus_3_scw_l1_condition
    pre_process_dict["vpu_3_para_clus_3_scw_l2_step"] = vpu_3_para_clus_3_scw_l2_step
    pre_process_dict["vpu_3_para_clus_3_scw_l2_condition"] = vpu_3_para_clus_3_scw_l2_condition
    pre_process_dict["vpu_3_para_clus_3_scw_l3_step"] = vpu_3_para_clus_3_scw_l3_step
    pre_process_dict["vpu_3_para_clus_3_scw_l3_condition"] = vpu_3_para_clus_3_scw_l3_condition
    pre_process_dict["vpu_3_para_clus_3_scw_l1_addr_step"] = vpu_3_para_clus_3_scw_l1_addr_step
    pre_process_dict["vpu_3_para_clus_3_scw_l2_addr_step"] = vpu_3_para_clus_3_scw_l2_addr_step
    pre_process_dict["vpu_3_para_clus_3_scw_l3_addr_step"] = vpu_3_para_clus_3_scw_l3_addr_step
    pre_process_dict["vpu_3_para_clus_3_scr_l1_step"] = vpu_3_para_clus_3_scr_l1_step
    pre_process_dict["vpu_3_para_clus_3_scr_l1_condition"] = vpu_3_para_clus_3_scr_l1_condition
    pre_process_dict["vpu_3_para_clus_3_scr_l2_step"] = vpu_3_para_clus_3_scr_l2_step
    pre_process_dict["vpu_3_para_clus_3_scr_l2_condition"] = vpu_3_para_clus_3_scr_l2_condition
    pre_process_dict["vpu_3_para_clus_3_scr_l3_step"] = vpu_3_para_clus_3_scr_l3_step
    pre_process_dict["vpu_3_para_clus_3_scr_l3_condition"] = vpu_3_para_clus_3_scr_l3_condition
    pre_process_dict["vpu_3_para_clus_3_scr_l1_addr_step"] = vpu_3_para_clus_3_scr_l1_addr_step
    pre_process_dict["vpu_3_para_clus_3_scr_l2_addr_step"] = vpu_3_para_clus_3_scr_l2_addr_step
    pre_process_dict["vpu_3_para_clus_3_scr_l3_addr_step"] = vpu_3_para_clus_3_scr_l3_addr_step
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_width_ratio"] = vpu_3_para_i_clus_3_resize_param_width_ratio
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_height_ratio"] = vpu_3_para_i_clus_3_resize_param_height_ratio
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_input_width"] = vpu_3_para_i_clus_3_resize_param_input_width
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_input_height"] = vpu_3_para_i_clus_3_resize_param_input_height
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_output_width"] = vpu_3_para_i_clus_3_resize_param_output_width
    pre_process_dict["vpu_3_para_i_clus_3_resize_param_output_height"] = vpu_3_para_i_clus_3_resize_param_output_height
    pre_process_dict["vpu_3_para_i_clus_3_pooling_param_input_width"] = vpu_3_para_i_clus_3_pooling_param_input_width
    pre_process_dict["vpu_3_para_i_clus_3_pooling_param_input_height"] = vpu_3_para_i_clus_3_pooling_param_input_height
    pre_process_dict["vpu_3_para_i_clus_3_pooling_param_output_width"] = vpu_3_para_i_clus_3_pooling_param_output_width
    pre_process_dict[
        "vpu_3_para_i_clus_3_pooling_param_output_height"] = vpu_3_para_i_clus_3_pooling_param_output_height
    pre_process_dict["vpu_3_para_i_clus_3_pooling_padding_mode"] = vpu_3_para_i_clus_3_pooling_padding_mode
    pre_process_dict["vpu_3_para_i_clus_3_pooling_padding_width"] = vpu_3_para_i_clus_3_pooling_padding_width
    pre_process_dict["vpu_3_para_i_clus_3_pooling_padding_height"] = vpu_3_para_i_clus_3_pooling_padding_height
    return pre_process_dict


def vpu_param(vpu_dict):
    register_dict = []
    # VPU_SF_RST
    VPU_SF_RST = []
    VPU_SF_RST.append(n_zeros_str(26))
    vpu_SF_para_vpu_fifo_group_sf_rst = vpu_dict["vpu_SF_para_vpu_fifo_group_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_vpu_fifo_group_sf_rst, 1))
    vpu_SF_para_global_line_buffer_sf_rst = vpu_dict["vpu_SF_para_global_line_buffer_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_global_line_buffer_sf_rst, 1))
    vpu_SF_para_sc_buffer_sf_rst = vpu_dict["vpu_SF_para_sc_buffer_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_sc_buffer_sf_rst, 1))
    vpu_SF_para_vpu_unit_sf_rst = vpu_dict["vpu_SF_para_vpu_unit_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_vpu_unit_sf_rst, 1))
    vpu_SF_para_interface_sf_rst = vpu_dict["vpu_SF_para_interface_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_interface_sf_rst, 1))
    vpu_SF_para_top_ctrl_sf_rst = vpu_dict["vpu_SF_para_top_ctrl_sf_rst"]
    VPU_SF_RST.append(intToBin(vpu_SF_para_top_ctrl_sf_rst, 1))
    VPU_SF_RST = bin_listTobin(VPU_SF_RST)
    VPU_SF_RST = binTohex(VPU_SF_RST, 32)
    register_dict.append(VPU_SF_RST)
    # VPU_TEST_MODE
    VPU_TEST_MODE = []
    VPU_TEST_MODE.append(n_zeros_str(11))
    vpu_TEST_para_done_len = vpu_dict["vpu_TEST_para_done_len"]
    VPU_TEST_MODE.append(intToBin(vpu_TEST_para_done_len, 12))
    vpu_TEST_para_bp_mode = vpu_dict["vpu_TEST_para_bp_mode"]
    VPU_TEST_MODE.append(intToBin(vpu_TEST_para_bp_mode, 2))
    vpu_TEST_para_vpu_unit_test_mode_sel = vpu_dict["vpu_TEST_para_vpu_unit_test_mode_sel"]
    VPU_TEST_MODE.append(intToBin(vpu_TEST_para_vpu_unit_test_mode_sel, 6))
    vpu_TEST_para_vpu_unit_test_mode_enable = vpu_dict["vpu_TEST_para_vpu_unit_test_mode_enable"]
    VPU_TEST_MODE.append(intToBin(vpu_TEST_para_vpu_unit_test_mode_enable, 1))
    VPU_TEST_MODE = bin_listTobin(VPU_TEST_MODE)
    VPU_TEST_MODE = binTohex(VPU_TEST_MODE, 32)
    register_dict.append(VPU_TEST_MODE)
    # VPU_SF_ENABLE
    VPU_SF_ENABLE = []
    VPU_SF_ENABLE.append(n_zeros_str(1))
    vpu_SF_para_odd_output_enable = vpu_dict["vpu_SF_para_odd_output_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_odd_output_enable, 1))
    vpu_SF_para_bp_mode_enable = vpu_dict["vpu_SF_para_bp_mode_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_bp_mode_enable, 1))
    vpu_SF_para_short_st_arb_enable = vpu_dict["vpu_SF_para_short_st_arb_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_short_st_arb_enable, 1))
    vpu_SF_para_psum_enable = vpu_dict["vpu_SF_para_psum_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_psum_enable, 1))
    vpu_SF_para_short_cut_buffer = vpu_dict["vpu_SF_para_short_cut_buffer"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_short_cut_buffer, 1))
    vpu_SF_para_line_controller_3 = vpu_dict["vpu_SF_para_line_controller_3"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_line_controller_3, 1))
    vpu_SF_para_line_controller_2 = vpu_dict["vpu_SF_para_line_controller_2"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_line_controller_2, 1))
    vpu_SF_para_line_controller_1 = vpu_dict["vpu_SF_para_line_controller_1"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_line_controller_1, 1))
    vpu_SF_para_line_controller_0 = vpu_dict["vpu_SF_para_line_controller_0"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_line_controller_0, 1))
    vpu_SF_para_fifo2line_buffer_enable = vpu_dict["vpu_SF_para_fifo2line_buffer_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_fifo2line_buffer_enable, 4))
    vpu_SF_para_global_line_buffer_enable = vpu_dict["vpu_SF_para_global_line_buffer_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_global_line_buffer_enable, 4))
    vpu_SF_para_input_fifo_enable = vpu_dict["vpu_SF_para_input_fifo_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_input_fifo_enable, 4))
    vpu_SF_para_vpu_sc_mode = vpu_dict["vpu_SF_para_vpu_sc_mode"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_vpu_sc_mode, 2))
    vpu_SF_para_vpu_sc_enable = vpu_dict["vpu_SF_para_vpu_sc_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_vpu_sc_enable, 4))
    vpu_SF_para_vpu_enable = vpu_dict["vpu_SF_para_vpu_enable"]
    VPU_SF_ENABLE.append(intToBin(vpu_SF_para_vpu_enable, 4))
    VPU_SF_ENABLE = bin_listTobin(VPU_SF_ENABLE)
    VPU_SF_ENABLE = binTohex(VPU_SF_ENABLE, 32)
    register_dict.append(VPU_SF_ENABLE)
    # VPU_TOP_CTRL
    VPU_TOP_CTRL = []
    VPU_TOP_CTRL.append(n_zeros_str(4))
    vpu_TOP_para_cim_weights_mode = vpu_dict["vpu_TOP_para_cim_weights_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_cim_weights_mode, 2))
    vpu_TOP_para_cluster_weights_mode = vpu_dict["vpu_TOP_para_cluster_weights_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_cluster_weights_mode, 2))
    vpu_TOP_para_interface_write_mode = vpu_dict["vpu_TOP_para_interface_write_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_interface_write_mode, 2))
    vpu_TOP_para_sc_width = vpu_dict["vpu_TOP_para_sc_width"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_sc_width, 9))
    vpu_TOP_para_switch_mode = vpu_dict["vpu_TOP_para_switch_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_switch_mode, 4))
    vpu_TOP_para_sc_mode = vpu_dict["vpu_TOP_para_sc_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_sc_mode, 3))
    vpu_TOP_para_line_buffer_mode = vpu_dict["vpu_TOP_para_line_buffer_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_line_buffer_mode, 2))
    vpu_TOP_para_fmt_channel_type = vpu_dict["vpu_TOP_para_fmt_channel_type"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_fmt_channel_type, 2))
    vpu_TOP_para_read_line_buffer_mode = vpu_dict["vpu_TOP_para_read_line_buffer_mode"]
    VPU_TOP_CTRL.append(intToBin(vpu_TOP_para_read_line_buffer_mode, 2))
    VPU_TOP_CTRL = bin_listTobin(VPU_TOP_CTRL)
    VPU_TOP_CTRL = binTohex(VPU_TOP_CTRL, 32)
    register_dict.append(VPU_TOP_CTRL)
    # ELEMENT_WISE_MODE
    ELEMENT_WISE_MODE = []
    ELEMENT_WISE_MODE.append(n_zeros_str(13))
    vpu_WISE_para_quantize_min = vpu_dict["vpu_WISE_para_quantize_min"]
    ELEMENT_WISE_MODE.append(intToBin(vpu_WISE_para_quantize_min, 8))
    vpu_WISE_para_quantize_max = vpu_dict["vpu_WISE_para_quantize_max"]
    ELEMENT_WISE_MODE.append(intToBin(vpu_WISE_para_quantize_max, 8))
    vpu_WISE_para_mode = vpu_dict["vpu_WISE_para_mode"]
    ELEMENT_WISE_MODE.append(intToBin(vpu_WISE_para_mode, 3))
    ELEMENT_WISE_MODE = bin_listTobin(ELEMENT_WISE_MODE)
    ELEMENT_WISE_MODE = binTohex(ELEMENT_WISE_MODE, 32)
    register_dict.append(ELEMENT_WISE_MODE)
    # ELEMENT_WISE_QUANTIZE_MUL
    ELEMENT_WISE_QUANTIZE_MUL = []
    vpu_WISE_para_quantize_mul = vpu_dict["vpu_WISE_para_quantize_mul"]
    ELEMENT_WISE_QUANTIZE_MUL.append(intToBin(vpu_WISE_para_quantize_mul, 32))
    ELEMENT_WISE_QUANTIZE_MUL = bin_listTobin(ELEMENT_WISE_QUANTIZE_MUL)
    ELEMENT_WISE_QUANTIZE_MUL = binTohex(ELEMENT_WISE_QUANTIZE_MUL, 32)
    register_dict.append(ELEMENT_WISE_QUANTIZE_MUL)
    # ELEMENT_WISE_QUANTIZE_SHIF
    ELEMENT_WISE_QUANTIZE_SHIF = []
    vpu_WISE_para_quantize_shf = vpu_dict["vpu_WISE_para_quantize_shf"]
    ELEMENT_WISE_QUANTIZE_SHIF.append(intToBin(vpu_WISE_para_quantize_shf, 32))
    ELEMENT_WISE_QUANTIZE_SHIF = bin_listTobin(ELEMENT_WISE_QUANTIZE_SHIF)
    ELEMENT_WISE_QUANTIZE_SHIF = binTohex(ELEMENT_WISE_QUANTIZE_SHIF, 32)
    register_dict.append(ELEMENT_WISE_QUANTIZE_SHIF)
    # ELEMENT_WISE_QUANTIZE_OFF
    ELEMENT_WISE_QUANTIZE_OFF = []
    vpu_WISE_para_quantize_off = vpu_dict["vpu_WISE_para_quantize_off"]
    ELEMENT_WISE_QUANTIZE_OFF.append(intToBin(vpu_WISE_para_quantize_off, 32))
    ELEMENT_WISE_QUANTIZE_OFF = bin_listTobin(ELEMENT_WISE_QUANTIZE_OFF)
    ELEMENT_WISE_QUANTIZE_OFF = binTohex(ELEMENT_WISE_QUANTIZE_OFF, 32)
    register_dict.append(ELEMENT_WISE_QUANTIZE_OFF)
    # ELEMENT_WISE_DEQUANTIZE_0_SCALE
    ELEMENT_WISE_DEQUANTIZE_0_SCALE = []
    vpu_WISE_para_element_wise_dequantize_0_sclale_o = vpu_dict["vpu_WISE_para_element_wise_dequantize_0_sclale_o"]
    ELEMENT_WISE_DEQUANTIZE_0_SCALE.append(intToBin(vpu_WISE_para_element_wise_dequantize_0_sclale_o, 32))
    ELEMENT_WISE_DEQUANTIZE_0_SCALE = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_0_SCALE)
    ELEMENT_WISE_DEQUANTIZE_0_SCALE = binTohex(ELEMENT_WISE_DEQUANTIZE_0_SCALE, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_0_SCALE)
    # ELEMENT_WISE_DEQUANTIZE_0_SHIFTER
    ELEMENT_WISE_DEQUANTIZE_0_SHIFTER = []
    vpu_WISE_para_element_wise_dequantize_0_shifter_o = vpu_dict["vpu_WISE_para_element_wise_dequantize_0_shifter_o"]
    ELEMENT_WISE_DEQUANTIZE_0_SHIFTER.append(intToBin(vpu_WISE_para_element_wise_dequantize_0_shifter_o, 32))
    ELEMENT_WISE_DEQUANTIZE_0_SHIFTER = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_0_SHIFTER)
    ELEMENT_WISE_DEQUANTIZE_0_SHIFTER = binTohex(ELEMENT_WISE_DEQUANTIZE_0_SHIFTER, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_0_SHIFTER)
    # ELEMENT_WISE_DEQUANTIZE_0_OFF
    ELEMENT_WISE_DEQUANTIZE_0_OFF = []
    vpu_WISE_para_dequantize_0_off = vpu_dict["vpu_WISE_para_dequantize_0_off"]
    ELEMENT_WISE_DEQUANTIZE_0_OFF.append(intToBin(vpu_WISE_para_dequantize_0_off, 32))
    ELEMENT_WISE_DEQUANTIZE_0_OFF = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_0_OFF)
    ELEMENT_WISE_DEQUANTIZE_0_OFF = binTohex(ELEMENT_WISE_DEQUANTIZE_0_OFF, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_0_OFF)
    # ELEMENT_WISE_DEQUANTIZE_1_SCALE
    ELEMENT_WISE_DEQUANTIZE_1_SCALE = []
    vpu_WISE_para_element_wise_dequantize_1_sclale_o = vpu_dict["vpu_WISE_para_element_wise_dequantize_1_sclale_o"]
    ELEMENT_WISE_DEQUANTIZE_1_SCALE.append(intToBin(vpu_WISE_para_element_wise_dequantize_1_sclale_o, 32))
    ELEMENT_WISE_DEQUANTIZE_1_SCALE = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_1_SCALE)
    ELEMENT_WISE_DEQUANTIZE_1_SCALE = binTohex(ELEMENT_WISE_DEQUANTIZE_1_SCALE, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_1_SCALE)
    # ELEMENT_WISE_DEQUANTIZE_1_SHIFTER
    ELEMENT_WISE_DEQUANTIZE_1_SHIFTER = []
    vpu_WISE_para_element_wise_dequantize_1_shifter_o = vpu_dict["vpu_WISE_para_element_wise_dequantize_1_shifter_o"]
    ELEMENT_WISE_DEQUANTIZE_1_SHIFTER.append(intToBin(vpu_WISE_para_element_wise_dequantize_1_shifter_o, 32))
    ELEMENT_WISE_DEQUANTIZE_1_SHIFTER = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_1_SHIFTER)
    ELEMENT_WISE_DEQUANTIZE_1_SHIFTER = binTohex(ELEMENT_WISE_DEQUANTIZE_1_SHIFTER, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_1_SHIFTER)
    # ELEMENT_WISE_DEQUANTIZE_1_OFF
    ELEMENT_WISE_DEQUANTIZE_1_OFF = []
    vpu_WISE_para_dequantize_1_off = vpu_dict["vpu_WISE_para_dequantize_1_off"]
    ELEMENT_WISE_DEQUANTIZE_1_OFF.append(intToBin(vpu_WISE_para_dequantize_1_off, 32))
    ELEMENT_WISE_DEQUANTIZE_1_OFF = bin_listTobin(ELEMENT_WISE_DEQUANTIZE_1_OFF)
    ELEMENT_WISE_DEQUANTIZE_1_OFF = binTohex(ELEMENT_WISE_DEQUANTIZE_1_OFF, 32)
    register_dict.append(ELEMENT_WISE_DEQUANTIZE_1_OFF)
    # ELEMENT_WISE_DIV_PARAM
    ELEMENT_WISE_DIV_PARAM = []
    ELEMENT_WISE_DIV_PARAM.append(n_zeros_str(12))
    vpu_WISE_para_div_fix_param = vpu_dict["vpu_WISE_para_div_fix_param"]
    ELEMENT_WISE_DIV_PARAM.append(intToBin(vpu_WISE_para_div_fix_param, 10))
    vpu_WISE_para_div_shifter = vpu_dict["vpu_WISE_para_div_shifter"]
    ELEMENT_WISE_DIV_PARAM.append(intToBin(vpu_WISE_para_div_shifter, 10))
    ELEMENT_WISE_DIV_PARAM = bin_listTobin(ELEMENT_WISE_DIV_PARAM)
    ELEMENT_WISE_DIV_PARAM = binTohex(ELEMENT_WISE_DIV_PARAM, 32)
    register_dict.append(ELEMENT_WISE_DIV_PARAM)
    # RS_BASIC_SETTING
    RS_BASIC_SETTING = []
    RS_BASIC_SETTING.append(n_zeros_str(30))
    vpu_BASIC_para_i_resize_param_half_pixal_flag = vpu_dict["vpu_BASIC_para_i_resize_param_half_pixal_flag"]
    RS_BASIC_SETTING.append(intToBin(vpu_BASIC_para_i_resize_param_half_pixal_flag, 1))
    vpu_BASIC_para_i_resize_param_bil_nn_sel_flag = vpu_dict["vpu_BASIC_para_i_resize_param_bil_nn_sel_flag"]
    RS_BASIC_SETTING.append(intToBin(vpu_BASIC_para_i_resize_param_bil_nn_sel_flag, 1))
    RS_BASIC_SETTING = bin_listTobin(RS_BASIC_SETTING)
    RS_BASIC_SETTING = binTohex(RS_BASIC_SETTING, 32)
    register_dict.append(RS_BASIC_SETTING)
    # PL_BASIC_SETTING
    PL_BASIC_SETTING = []
    PL_BASIC_SETTING.append(n_zeros_str(29))
    vpu_BASIC_para_pl_func_mode = vpu_dict["vpu_BASIC_para_pl_func_mode"]
    PL_BASIC_SETTING.append(intToBin(vpu_BASIC_para_pl_func_mode, 1))
    vpu_BASIC_para_pl_factor = vpu_dict["vpu_BASIC_para_pl_factor"]
    PL_BASIC_SETTING.append(intToBin(vpu_BASIC_para_pl_factor, 2))
    PL_BASIC_SETTING = bin_listTobin(PL_BASIC_SETTING)
    PL_BASIC_SETTING = binTohex(PL_BASIC_SETTING, 32)
    register_dict.append(PL_BASIC_SETTING)
    # PL_FILTER_WH
    PL_FILTER_WH = []
    PL_FILTER_WH.append(n_zeros_str(10))
    vpu_FILTER_para_i_pooling_filter_width = vpu_dict["vpu_FILTER_para_i_pooling_filter_width"]
    PL_FILTER_WH.append(intToBin(vpu_FILTER_para_i_pooling_filter_width, 11))
    vpu_FILTER_para_i_pooling_filter_height = vpu_dict["vpu_FILTER_para_i_pooling_filter_height"]
    PL_FILTER_WH.append(intToBin(vpu_FILTER_para_i_pooling_filter_height, 11))
    PL_FILTER_WH = bin_listTobin(PL_FILTER_WH)
    PL_FILTER_WH = binTohex(PL_FILTER_WH, 32)
    register_dict.append(PL_FILTER_WH)
    # PL_STRIDE_WH
    PL_STRIDE_WH = []
    PL_STRIDE_WH.append(n_zeros_str(10))
    vpu_STRIDE_para_i_pooling_stride_width = vpu_dict["vpu_STRIDE_para_i_pooling_stride_width"]
    PL_STRIDE_WH.append(intToBin(vpu_STRIDE_para_i_pooling_stride_width, 11))
    vpu_STRIDE_para_i_pooling_stride_height = vpu_dict["vpu_STRIDE_para_i_pooling_stride_height"]
    PL_STRIDE_WH.append(intToBin(vpu_STRIDE_para_i_pooling_stride_height, 11))
    PL_STRIDE_WH = bin_listTobin(PL_STRIDE_WH)
    PL_STRIDE_WH = binTohex(PL_STRIDE_WH, 32)
    register_dict.append(PL_STRIDE_WH)
    # PLRS_BASIC_COMMON_SETTING
    PLRS_BASIC_COMMON_SETTING = []
    PLRS_BASIC_COMMON_SETTING.append(n_zeros_str(21))
    vpu_BASIC_para_i_fmt_width = vpu_dict["vpu_BASIC_para_i_fmt_width"]
    PLRS_BASIC_COMMON_SETTING.append(intToBin(vpu_BASIC_para_i_fmt_width, 10))
    vpu_BASIC_para_i_pl_rs_sel = vpu_dict["vpu_BASIC_para_i_pl_rs_sel"]
    PLRS_BASIC_COMMON_SETTING.append(intToBin(vpu_BASIC_para_i_pl_rs_sel, 1))
    PLRS_BASIC_COMMON_SETTING = bin_listTobin(PLRS_BASIC_COMMON_SETTING)
    PLRS_BASIC_COMMON_SETTING = binTohex(PLRS_BASIC_COMMON_SETTING, 32)
    register_dict.append(PLRS_BASIC_COMMON_SETTING)
    # VPU_INTERFACE_B0_AD
    VPU_INTERFACE_B0_AD = []
    VPU_INTERFACE_B0_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b0_ad = vpu_dict["vpu_INTERFACE_para_b0_ad"]
    VPU_INTERFACE_B0_AD.append(intToBin(vpu_INTERFACE_para_b0_ad, 19))
    VPU_INTERFACE_B0_AD = bin_listTobin(VPU_INTERFACE_B0_AD)
    VPU_INTERFACE_B0_AD = binTohex(VPU_INTERFACE_B0_AD, 32)
    register_dict.append(VPU_INTERFACE_B0_AD)
    # VPU_INTERFACE_B1_AD
    VPU_INTERFACE_B1_AD = []
    VPU_INTERFACE_B1_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b1_ad = vpu_dict["vpu_INTERFACE_para_b1_ad"]
    VPU_INTERFACE_B1_AD.append(intToBin(vpu_INTERFACE_para_b1_ad, 19))
    VPU_INTERFACE_B1_AD = bin_listTobin(VPU_INTERFACE_B1_AD)
    VPU_INTERFACE_B1_AD = binTohex(VPU_INTERFACE_B1_AD, 32)
    register_dict.append(VPU_INTERFACE_B1_AD)
    # VPU_INTERFACE_B2_AD
    VPU_INTERFACE_B2_AD = []
    VPU_INTERFACE_B2_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b2_ad = vpu_dict["vpu_INTERFACE_para_b2_ad"]
    VPU_INTERFACE_B2_AD.append(intToBin(vpu_INTERFACE_para_b2_ad, 19))
    VPU_INTERFACE_B2_AD = bin_listTobin(VPU_INTERFACE_B2_AD)
    VPU_INTERFACE_B2_AD = binTohex(VPU_INTERFACE_B2_AD, 32)
    register_dict.append(VPU_INTERFACE_B2_AD)
    # VPU_INTERFACE_B3_AD
    VPU_INTERFACE_B3_AD = []
    VPU_INTERFACE_B3_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b3_ad = vpu_dict["vpu_INTERFACE_para_b3_ad"]
    VPU_INTERFACE_B3_AD.append(intToBin(vpu_INTERFACE_para_b3_ad, 19))
    VPU_INTERFACE_B3_AD = bin_listTobin(VPU_INTERFACE_B3_AD)
    VPU_INTERFACE_B3_AD = binTohex(VPU_INTERFACE_B3_AD, 32)
    register_dict.append(VPU_INTERFACE_B3_AD)
    # VPU_INTERFACE_B4_WT_AD
    VPU_INTERFACE_B4_WT_AD = []
    VPU_INTERFACE_B4_WT_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b4_wt_addr = vpu_dict["vpu_INTERFACE_para_b4_wt_addr"]
    VPU_INTERFACE_B4_WT_AD.append(intToBin(vpu_INTERFACE_para_b4_wt_addr, 19))
    VPU_INTERFACE_B4_WT_AD = bin_listTobin(VPU_INTERFACE_B4_WT_AD)
    VPU_INTERFACE_B4_WT_AD = binTohex(VPU_INTERFACE_B4_WT_AD, 32)
    register_dict.append(VPU_INTERFACE_B4_WT_AD)
    # VPU_INTERFACE_B5_WT_AD
    VPU_INTERFACE_B5_WT_AD = []
    VPU_INTERFACE_B5_WT_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b5_wt_addr = vpu_dict["vpu_INTERFACE_para_b5_wt_addr"]
    VPU_INTERFACE_B5_WT_AD.append(intToBin(vpu_INTERFACE_para_b5_wt_addr, 19))
    VPU_INTERFACE_B5_WT_AD = bin_listTobin(VPU_INTERFACE_B5_WT_AD)
    VPU_INTERFACE_B5_WT_AD = binTohex(VPU_INTERFACE_B5_WT_AD, 32)
    register_dict.append(VPU_INTERFACE_B5_WT_AD)
    # VPU_INTERFACE_B6_WT_AD
    VPU_INTERFACE_B6_WT_AD = []
    VPU_INTERFACE_B6_WT_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b6_wt_addr = vpu_dict["vpu_INTERFACE_para_b6_wt_addr"]
    VPU_INTERFACE_B6_WT_AD.append(intToBin(vpu_INTERFACE_para_b6_wt_addr, 19))
    VPU_INTERFACE_B6_WT_AD = bin_listTobin(VPU_INTERFACE_B6_WT_AD)
    VPU_INTERFACE_B6_WT_AD = binTohex(VPU_INTERFACE_B6_WT_AD, 32)
    register_dict.append(VPU_INTERFACE_B6_WT_AD)
    # VPU_INTERFACE_B7_WT_AD
    VPU_INTERFACE_B7_WT_AD = []
    VPU_INTERFACE_B7_WT_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b7_wt_addr = vpu_dict["vpu_INTERFACE_para_b7_wt_addr"]
    VPU_INTERFACE_B7_WT_AD.append(intToBin(vpu_INTERFACE_para_b7_wt_addr, 19))
    VPU_INTERFACE_B7_WT_AD = bin_listTobin(VPU_INTERFACE_B7_WT_AD)
    VPU_INTERFACE_B7_WT_AD = binTohex(VPU_INTERFACE_B7_WT_AD, 32)
    register_dict.append(VPU_INTERFACE_B7_WT_AD)
    # VPU_INTERFACE_B4_RD_AD
    VPU_INTERFACE_B4_RD_AD = []
    VPU_INTERFACE_B4_RD_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b4_rd_addr = vpu_dict["vpu_INTERFACE_para_b4_rd_addr"]
    VPU_INTERFACE_B4_RD_AD.append(intToBin(vpu_INTERFACE_para_b4_rd_addr, 19))
    VPU_INTERFACE_B4_RD_AD = bin_listTobin(VPU_INTERFACE_B4_RD_AD)
    VPU_INTERFACE_B4_RD_AD = binTohex(VPU_INTERFACE_B4_RD_AD, 32)
    register_dict.append(VPU_INTERFACE_B4_RD_AD)
    # VPU_INTERFACE_B5_RD_AD
    VPU_INTERFACE_B5_RD_AD = []
    VPU_INTERFACE_B5_RD_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b5_rd_addr = vpu_dict["vpu_INTERFACE_para_b5_rd_addr"]
    VPU_INTERFACE_B5_RD_AD.append(intToBin(vpu_INTERFACE_para_b5_rd_addr, 19))
    VPU_INTERFACE_B5_RD_AD = bin_listTobin(VPU_INTERFACE_B5_RD_AD)
    VPU_INTERFACE_B5_RD_AD = binTohex(VPU_INTERFACE_B5_RD_AD, 32)
    register_dict.append(VPU_INTERFACE_B5_RD_AD)
    # VPU_INTERFACE_B6_RD_AD
    VPU_INTERFACE_B6_RD_AD = []
    VPU_INTERFACE_B6_RD_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b6_rd_addr = vpu_dict["vpu_INTERFACE_para_b6_rd_addr"]
    VPU_INTERFACE_B6_RD_AD.append(intToBin(vpu_INTERFACE_para_b6_rd_addr, 19))
    VPU_INTERFACE_B6_RD_AD = bin_listTobin(VPU_INTERFACE_B6_RD_AD)
    VPU_INTERFACE_B6_RD_AD = binTohex(VPU_INTERFACE_B6_RD_AD, 32)
    register_dict.append(VPU_INTERFACE_B6_RD_AD)
    # VPU_INTERFACE_B7_RD_AD
    VPU_INTERFACE_B7_RD_AD = []
    VPU_INTERFACE_B7_RD_AD.append(n_zeros_str(13))
    vpu_INTERFACE_para_b7_rd_addr = vpu_dict["vpu_INTERFACE_para_b7_rd_addr"]
    VPU_INTERFACE_B7_RD_AD.append(intToBin(vpu_INTERFACE_para_b7_rd_addr, 19))
    VPU_INTERFACE_B7_RD_AD = bin_listTobin(VPU_INTERFACE_B7_RD_AD)
    VPU_INTERFACE_B7_RD_AD = binTohex(VPU_INTERFACE_B7_RD_AD, 32)
    register_dict.append(VPU_INTERFACE_B7_RD_AD)
    # VPU_GLOBAL_BUFFER_SRAM_PARAM
    VPU_GLOBAL_BUFFER_SRAM_PARAM = []
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(n_zeros_str(2))
    vpu_GLOBAL_para_sc_buffer_stov = vpu_dict["vpu_GLOBAL_para_sc_buffer_stov"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_stov, 1))
    vpu_GLOBAL_para_sc_buffer_ema = vpu_dict["vpu_GLOBAL_para_sc_buffer_ema"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_ema, 3))
    vpu_GLOBAL_para_sc_buffer_emaw = vpu_dict["vpu_GLOBAL_para_sc_buffer_emaw"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_emaw, 2))
    vpu_GLOBAL_para_sc_buffer_emas = vpu_dict["vpu_GLOBAL_para_sc_buffer_emas"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_emas, 1))
    vpu_GLOBAL_para_sc_buffer_ret1n = vpu_dict["vpu_GLOBAL_para_sc_buffer_ret1n"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_ret1n, 1))
    vpu_GLOBAL_para_sc_buffer_rawl = vpu_dict["vpu_GLOBAL_para_sc_buffer_rawl"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_rawl, 1))
    vpu_GLOBAL_para_sc_buffer_rawlm = vpu_dict["vpu_GLOBAL_para_sc_buffer_rawlm"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_rawlm, 2))
    vpu_GLOBAL_para_sc_buffer_wabl = vpu_dict["vpu_GLOBAL_para_sc_buffer_wabl"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_wabl, 1))
    vpu_GLOBAL_para_sc_buffer_wablm = vpu_dict["vpu_GLOBAL_para_sc_buffer_wablm"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_sc_buffer_wablm, 3))
    vpu_GLOBAL_para_global_buffer_stov = vpu_dict["vpu_GLOBAL_para_global_buffer_stov"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_stov, 1))
    vpu_GLOBAL_para_global_buffer_ema = vpu_dict["vpu_GLOBAL_para_global_buffer_ema"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_ema, 3))
    vpu_GLOBAL_para_global_buffer_emaw = vpu_dict["vpu_GLOBAL_para_global_buffer_emaw"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_emaw, 2))
    vpu_GLOBAL_para_global_buffer_emas = vpu_dict["vpu_GLOBAL_para_global_buffer_emas"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_emas, 1))
    vpu_GLOBAL_para_global_buffer_ret1n = vpu_dict["vpu_GLOBAL_para_global_buffer_ret1n"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_ret1n, 1))
    vpu_GLOBAL_para_global_buffer_rawl = vpu_dict["vpu_GLOBAL_para_global_buffer_rawl"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_rawl, 1))
    vpu_GLOBAL_para_global_buffer_rawlm = vpu_dict["vpu_GLOBAL_para_global_buffer_rawlm"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_rawlm, 2))
    vpu_GLOBAL_para_global_buffer_wabl = vpu_dict["vpu_GLOBAL_para_global_buffer_wabl"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_wabl, 1))
    vpu_GLOBAL_para_global_buffer_wablm = vpu_dict["vpu_GLOBAL_para_global_buffer_wablm"]
    VPU_GLOBAL_BUFFER_SRAM_PARAM.append(intToBin(vpu_GLOBAL_para_global_buffer_wablm, 3))
    VPU_GLOBAL_BUFFER_SRAM_PARAM = bin_listTobin(VPU_GLOBAL_BUFFER_SRAM_PARAM)
    VPU_GLOBAL_BUFFER_SRAM_PARAM = binTohex(VPU_GLOBAL_BUFFER_SRAM_PARAM, 32)
    register_dict.append(VPU_GLOBAL_BUFFER_SRAM_PARAM)
    # VPU_LINE_BUFFER_SRAM_PARAM
    VPU_LINE_BUFFER_SRAM_PARAM = []
    VPU_LINE_BUFFER_SRAM_PARAM.append(n_zeros_str(17))
    vpu_LINE_para_line_buffer_stov = vpu_dict["vpu_LINE_para_line_buffer_stov"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_stov, 1))
    vpu_LINE_para_line_buffer_ema = vpu_dict["vpu_LINE_para_line_buffer_ema"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_ema, 3))
    vpu_LINE_para_line_buffer_emaw = vpu_dict["vpu_LINE_para_line_buffer_emaw"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_emaw, 2))
    vpu_LINE_para_line_buffer_emas = vpu_dict["vpu_LINE_para_line_buffer_emas"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_emas, 1))
    vpu_LINE_para_line_buffer_ret1n = vpu_dict["vpu_LINE_para_line_buffer_ret1n"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_ret1n, 1))
    vpu_LINE_para_line_buffer_rawl = vpu_dict["vpu_LINE_para_line_buffer_rawl"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_rawl, 1))
    vpu_LINE_para_line_buffer_rawlm = vpu_dict["vpu_LINE_para_line_buffer_rawlm"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_rawlm, 2))
    vpu_LINE_para_line_buffer_wabl = vpu_dict["vpu_LINE_para_line_buffer_wabl"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_wabl, 1))
    vpu_LINE_para_line_buffer_wablm = vpu_dict["vpu_LINE_para_line_buffer_wablm"]
    VPU_LINE_BUFFER_SRAM_PARAM.append(intToBin(vpu_LINE_para_line_buffer_wablm, 3))
    VPU_LINE_BUFFER_SRAM_PARAM = bin_listTobin(VPU_LINE_BUFFER_SRAM_PARAM)
    VPU_LINE_BUFFER_SRAM_PARAM = binTohex(VPU_LINE_BUFFER_SRAM_PARAM, 32)
    register_dict.append(VPU_LINE_BUFFER_SRAM_PARAM)
    # VPU_LUT_SRAM_PARAM
    VPU_LUT_SRAM_PARAM = []
    VPU_LUT_SRAM_PARAM.append(n_zeros_str(17))
    vpu_LUT_para_lut_stov = vpu_dict["vpu_LUT_para_lut_stov"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_stov, 1))
    vpu_LUT_para_lut_ema = vpu_dict["vpu_LUT_para_lut_ema"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_ema, 3))
    vpu_LUT_para_lut_emaw = vpu_dict["vpu_LUT_para_lut_emaw"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_emaw, 2))
    vpu_LUT_para_lut_emas = vpu_dict["vpu_LUT_para_lut_emas"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_emas, 1))
    vpu_LUT_para_lut_ret1n = vpu_dict["vpu_LUT_para_lut_ret1n"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_ret1n, 1))
    vpu_LUT_para_lut_rawl = vpu_dict["vpu_LUT_para_lut_rawl"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_rawl, 1))
    vpu_LUT_para_lut_rawlm = vpu_dict["vpu_LUT_para_lut_rawlm"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_rawlm, 2))
    vpu_LUT_para_lut_wabl = vpu_dict["vpu_LUT_para_lut_wabl"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_wabl, 1))
    vpu_LUT_para_lut_wablm = vpu_dict["vpu_LUT_para_lut_wablm"]
    VPU_LUT_SRAM_PARAM.append(intToBin(vpu_LUT_para_lut_wablm, 3))
    VPU_LUT_SRAM_PARAM = bin_listTobin(VPU_LUT_SRAM_PARAM)
    VPU_LUT_SRAM_PARAM = binTohex(VPU_LUT_SRAM_PARAM, 32)
    register_dict.append(VPU_LUT_SRAM_PARAM)
    # VPU_UNIT_SEED
    VPU_UNIT_SEED = []
    vpu_UNIT_para_i_vpu_unit_seed = vpu_dict["vpu_UNIT_para_i_vpu_unit_seed"]
    VPU_UNIT_SEED.append(intToBin(vpu_UNIT_para_i_vpu_unit_seed, 32))
    VPU_UNIT_SEED = bin_listTobin(VPU_UNIT_SEED)
    VPU_UNIT_SEED = binTohex(VPU_UNIT_SEED, 32)
    register_dict.append(VPU_UNIT_SEED)
    # VPU_INTERFACE_SEED
    VPU_INTERFACE_SEED = []
    vpu_INTERFACE_para_i_vpu_interface_seed = vpu_dict["vpu_INTERFACE_para_i_vpu_interface_seed"]
    VPU_INTERFACE_SEED.append(intToBin(vpu_INTERFACE_para_i_vpu_interface_seed, 32))
    VPU_INTERFACE_SEED = bin_listTobin(VPU_INTERFACE_SEED)
    VPU_INTERFACE_SEED = binTohex(VPU_INTERFACE_SEED, 32)
    register_dict.append(VPU_INTERFACE_SEED)
    # VPU_IN_FIFO_GROUP_SEED
    VPU_IN_FIFO_GROUP_SEED = []
    vpu_IN_para_i_vpu_in_fifo_group_seed = vpu_dict["vpu_IN_para_i_vpu_in_fifo_group_seed"]
    VPU_IN_FIFO_GROUP_SEED.append(intToBin(vpu_IN_para_i_vpu_in_fifo_group_seed, 32))
    VPU_IN_FIFO_GROUP_SEED = bin_listTobin(VPU_IN_FIFO_GROUP_SEED)
    VPU_IN_FIFO_GROUP_SEED = binTohex(VPU_IN_FIFO_GROUP_SEED, 32)
    register_dict.append(VPU_IN_FIFO_GROUP_SEED)
    # TARGET_DATA_AMOUNT
    TARGET_DATA_AMOUNT = []
    vpu_DATA_para_i_target_data_amount = vpu_dict["vpu_DATA_para_i_target_data_amount"]
    TARGET_DATA_AMOUNT.append(intToBin(vpu_DATA_para_i_target_data_amount, 32))
    TARGET_DATA_AMOUNT = bin_listTobin(TARGET_DATA_AMOUNT)
    TARGET_DATA_AMOUNT = binTohex(TARGET_DATA_AMOUNT, 32)
    register_dict.append(TARGET_DATA_AMOUNT)
    # VPU_BIST_VPU_UNIT_OUT_DATA_0
    VPU_BIST_VPU_UNIT_OUT_DATA_0 = []
    vpu_BIST_para_o_ans_data = vpu_dict["vpu_BIST_para_o_ans_data"]
    VPU_BIST_VPU_UNIT_OUT_DATA_0.append(intToBin(vpu_BIST_para_o_ans_data, 32))
    VPU_BIST_VPU_UNIT_OUT_DATA_0 = bin_listTobin(VPU_BIST_VPU_UNIT_OUT_DATA_0)
    VPU_BIST_VPU_UNIT_OUT_DATA_0 = binTohex(VPU_BIST_VPU_UNIT_OUT_DATA_0, 32)
    register_dict.append(VPU_BIST_VPU_UNIT_OUT_DATA_0)
    # VPU_BIST_VPU_UNIT_OUT_DATA_1
    VPU_BIST_VPU_UNIT_OUT_DATA_1 = []
    vpu_BIST_para_o_ans_data_sc = vpu_dict["vpu_BIST_para_o_ans_data_sc"]
    VPU_BIST_VPU_UNIT_OUT_DATA_1.append(intToBin(vpu_BIST_para_o_ans_data_sc, 32))
    VPU_BIST_VPU_UNIT_OUT_DATA_1 = bin_listTobin(VPU_BIST_VPU_UNIT_OUT_DATA_1)
    VPU_BIST_VPU_UNIT_OUT_DATA_1 = binTohex(VPU_BIST_VPU_UNIT_OUT_DATA_1, 32)
    register_dict.append(VPU_BIST_VPU_UNIT_OUT_DATA_1)
    # VPU_BIST_VPU_UNIT_OUT_DATA_2
    VPU_BIST_VPU_UNIT_OUT_DATA_2 = []
    vpu_BIST_para_o_ans_data_wo_q = vpu_dict["vpu_BIST_para_o_ans_data_wo_q"]
    VPU_BIST_VPU_UNIT_OUT_DATA_2.append(intToBin(vpu_BIST_para_o_ans_data_wo_q, 32))
    VPU_BIST_VPU_UNIT_OUT_DATA_2 = bin_listTobin(VPU_BIST_VPU_UNIT_OUT_DATA_2)
    VPU_BIST_VPU_UNIT_OUT_DATA_2 = binTohex(VPU_BIST_VPU_UNIT_OUT_DATA_2, 32)
    register_dict.append(VPU_BIST_VPU_UNIT_OUT_DATA_2)
    # VPU_BIST_VPU_UNIT_OUT_DATA_3
    VPU_BIST_VPU_UNIT_OUT_DATA_3 = []
    VPU_BIST_VPU_UNIT_OUT_DATA_3.append(n_zeros_str(30))
    vpu_BIST_para_o_ans_done_flag = vpu_dict["vpu_BIST_para_o_ans_done_flag"]
    VPU_BIST_VPU_UNIT_OUT_DATA_3.append(intToBin(vpu_BIST_para_o_ans_done_flag, 1))
    vpu_BIST_para_o_ans_col_done = vpu_dict["vpu_BIST_para_o_ans_col_done"]
    VPU_BIST_VPU_UNIT_OUT_DATA_3.append(intToBin(vpu_BIST_para_o_ans_col_done, 1))
    VPU_BIST_VPU_UNIT_OUT_DATA_3 = bin_listTobin(VPU_BIST_VPU_UNIT_OUT_DATA_3)
    VPU_BIST_VPU_UNIT_OUT_DATA_3 = binTohex(VPU_BIST_VPU_UNIT_OUT_DATA_3, 32)
    register_dict.append(VPU_BIST_VPU_UNIT_OUT_DATA_3)
    # VPU_BIST_INPUT_FIFO_PSUM
    VPU_BIST_INPUT_FIFO_PSUM = []
    vpu_BIST_para_all_ans_data_out_psum = vpu_dict["vpu_BIST_para_all_ans_data_out_psum"]
    VPU_BIST_INPUT_FIFO_PSUM.append(intToBin(vpu_BIST_para_all_ans_data_out_psum, 32))
    VPU_BIST_INPUT_FIFO_PSUM = bin_listTobin(VPU_BIST_INPUT_FIFO_PSUM)
    VPU_BIST_INPUT_FIFO_PSUM = binTohex(VPU_BIST_INPUT_FIFO_PSUM, 32)
    register_dict.append(VPU_BIST_INPUT_FIFO_PSUM)
    # VPU_BIST_INPUT_FIFO_0
    VPU_BIST_INPUT_FIFO_0 = []
    vpu_BIST_para_all_ans_data_out_0 = vpu_dict["vpu_BIST_para_all_ans_data_out_0"]
    VPU_BIST_INPUT_FIFO_0.append(intToBin(vpu_BIST_para_all_ans_data_out_0, 32))
    VPU_BIST_INPUT_FIFO_0 = bin_listTobin(VPU_BIST_INPUT_FIFO_0)
    VPU_BIST_INPUT_FIFO_0 = binTohex(VPU_BIST_INPUT_FIFO_0, 32)
    register_dict.append(VPU_BIST_INPUT_FIFO_0)
    # VPU_BIST_INPUT_FIFO_1
    VPU_BIST_INPUT_FIFO_1 = []
    vpu_BIST_para_all_ans_data_out_1 = vpu_dict["vpu_BIST_para_all_ans_data_out_1"]
    VPU_BIST_INPUT_FIFO_1.append(intToBin(vpu_BIST_para_all_ans_data_out_1, 32))
    VPU_BIST_INPUT_FIFO_1 = bin_listTobin(VPU_BIST_INPUT_FIFO_1)
    VPU_BIST_INPUT_FIFO_1 = binTohex(VPU_BIST_INPUT_FIFO_1, 32)
    register_dict.append(VPU_BIST_INPUT_FIFO_1)
    # VPU_BIST_INPUT_FIFO_2
    VPU_BIST_INPUT_FIFO_2 = []
    vpu_BIST_para_all_ans_data_out_2 = vpu_dict["vpu_BIST_para_all_ans_data_out_2"]
    VPU_BIST_INPUT_FIFO_2.append(intToBin(vpu_BIST_para_all_ans_data_out_2, 32))
    VPU_BIST_INPUT_FIFO_2 = bin_listTobin(VPU_BIST_INPUT_FIFO_2)
    VPU_BIST_INPUT_FIFO_2 = binTohex(VPU_BIST_INPUT_FIFO_2, 32)
    register_dict.append(VPU_BIST_INPUT_FIFO_2)
    # VPU_BIST_INPUT_FIFO_3
    VPU_BIST_INPUT_FIFO_3 = []
    vpu_BIST_para_all_ans_data_out_3 = vpu_dict["vpu_BIST_para_all_ans_data_out_3"]
    VPU_BIST_INPUT_FIFO_3.append(intToBin(vpu_BIST_para_all_ans_data_out_3, 32))
    VPU_BIST_INPUT_FIFO_3 = bin_listTobin(VPU_BIST_INPUT_FIFO_3)
    VPU_BIST_INPUT_FIFO_3 = binTohex(VPU_BIST_INPUT_FIFO_3, 32)
    register_dict.append(VPU_BIST_INPUT_FIFO_3)
    # VPU_BIST_INTERFACE_B0_DATA
    VPU_BIST_INTERFACE_B0_DATA = []
    vpu_BIST_para_o_b0_ans_data = vpu_dict["vpu_BIST_para_o_b0_ans_data"]
    VPU_BIST_INTERFACE_B0_DATA.append(intToBin(vpu_BIST_para_o_b0_ans_data, 32))
    VPU_BIST_INTERFACE_B0_DATA = bin_listTobin(VPU_BIST_INTERFACE_B0_DATA)
    VPU_BIST_INTERFACE_B0_DATA = binTohex(VPU_BIST_INTERFACE_B0_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B0_DATA)
    # VPU_BIST_INTERFACE_B0_ADDR
    VPU_BIST_INTERFACE_B0_ADDR = []
    VPU_BIST_INTERFACE_B0_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b0_ans_ctrl = vpu_dict["vpu_BIST_para_o_b0_ans_ctrl"]
    VPU_BIST_INTERFACE_B0_ADDR.append(intToBin(vpu_BIST_para_o_b0_ans_ctrl, 3))
    vpu_BIST_para_o_b0_ans_addr = vpu_dict["vpu_BIST_para_o_b0_ans_addr"]
    VPU_BIST_INTERFACE_B0_ADDR.append(intToBin(vpu_BIST_para_o_b0_ans_addr, 21))
    VPU_BIST_INTERFACE_B0_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B0_ADDR)
    VPU_BIST_INTERFACE_B0_ADDR = binTohex(VPU_BIST_INTERFACE_B0_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B0_ADDR)
    # VPU_BIST_INTERFACE_B1_DATA
    VPU_BIST_INTERFACE_B1_DATA = []
    vpu_BIST_para_o_b1_ans_data = vpu_dict["vpu_BIST_para_o_b1_ans_data"]
    VPU_BIST_INTERFACE_B1_DATA.append(intToBin(vpu_BIST_para_o_b1_ans_data, 32))
    VPU_BIST_INTERFACE_B1_DATA = bin_listTobin(VPU_BIST_INTERFACE_B1_DATA)
    VPU_BIST_INTERFACE_B1_DATA = binTohex(VPU_BIST_INTERFACE_B1_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B1_DATA)
    # VPU_BIST_INTERFACE_B1_ADDR
    VPU_BIST_INTERFACE_B1_ADDR = []
    VPU_BIST_INTERFACE_B1_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b1_ans_ctrl = vpu_dict["vpu_BIST_para_o_b1_ans_ctrl"]
    VPU_BIST_INTERFACE_B1_ADDR.append(intToBin(vpu_BIST_para_o_b1_ans_ctrl, 3))
    vpu_BIST_para_o_b1_ans_addr = vpu_dict["vpu_BIST_para_o_b1_ans_addr"]
    VPU_BIST_INTERFACE_B1_ADDR.append(intToBin(vpu_BIST_para_o_b1_ans_addr, 21))
    VPU_BIST_INTERFACE_B1_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B1_ADDR)
    VPU_BIST_INTERFACE_B1_ADDR = binTohex(VPU_BIST_INTERFACE_B1_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B1_ADDR)
    # VPU_BIST_INTERFACE_B2_DATA
    VPU_BIST_INTERFACE_B2_DATA = []
    vpu_BIST_para_o_b2_ans_data = vpu_dict["vpu_BIST_para_o_b2_ans_data"]
    VPU_BIST_INTERFACE_B2_DATA.append(intToBin(vpu_BIST_para_o_b2_ans_data, 32))
    VPU_BIST_INTERFACE_B2_DATA = bin_listTobin(VPU_BIST_INTERFACE_B2_DATA)
    VPU_BIST_INTERFACE_B2_DATA = binTohex(VPU_BIST_INTERFACE_B2_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B2_DATA)
    # VPU_BIST_INTERFACE_B2_ADDR
    VPU_BIST_INTERFACE_B2_ADDR = []
    VPU_BIST_INTERFACE_B2_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b2_ans_ctrl = vpu_dict["vpu_BIST_para_o_b2_ans_ctrl"]
    VPU_BIST_INTERFACE_B2_ADDR.append(intToBin(vpu_BIST_para_o_b2_ans_ctrl, 3))
    vpu_BIST_para_o_b2_ans_addr = vpu_dict["vpu_BIST_para_o_b2_ans_addr"]
    VPU_BIST_INTERFACE_B2_ADDR.append(intToBin(vpu_BIST_para_o_b2_ans_addr, 21))
    VPU_BIST_INTERFACE_B2_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B2_ADDR)
    VPU_BIST_INTERFACE_B2_ADDR = binTohex(VPU_BIST_INTERFACE_B2_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B2_ADDR)
    # VPU_BIST_INTERFACE_B3_DATA
    VPU_BIST_INTERFACE_B3_DATA = []
    vpu_BIST_para_o_b3_ans_data = vpu_dict["vpu_BIST_para_o_b3_ans_data"]
    VPU_BIST_INTERFACE_B3_DATA.append(intToBin(vpu_BIST_para_o_b3_ans_data, 32))
    VPU_BIST_INTERFACE_B3_DATA = bin_listTobin(VPU_BIST_INTERFACE_B3_DATA)
    VPU_BIST_INTERFACE_B3_DATA = binTohex(VPU_BIST_INTERFACE_B3_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B3_DATA)
    # VPU_BIST_INTERFACE_B3_ADDR
    VPU_BIST_INTERFACE_B3_ADDR = []
    VPU_BIST_INTERFACE_B3_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b3_ans_ctrl = vpu_dict["vpu_BIST_para_o_b3_ans_ctrl"]
    VPU_BIST_INTERFACE_B3_ADDR.append(intToBin(vpu_BIST_para_o_b3_ans_ctrl, 3))
    vpu_BIST_para_o_b3_ans_addr = vpu_dict["vpu_BIST_para_o_b3_ans_addr"]
    VPU_BIST_INTERFACE_B3_ADDR.append(intToBin(vpu_BIST_para_o_b3_ans_addr, 21))
    VPU_BIST_INTERFACE_B3_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B3_ADDR)
    VPU_BIST_INTERFACE_B3_ADDR = binTohex(VPU_BIST_INTERFACE_B3_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B3_ADDR)
    # VPU_BIST_INTERFACE_B4_DATA
    VPU_BIST_INTERFACE_B4_DATA = []
    vpu_BIST_para_o_b4_ans_data = vpu_dict["vpu_BIST_para_o_b4_ans_data"]
    VPU_BIST_INTERFACE_B4_DATA.append(intToBin(vpu_BIST_para_o_b4_ans_data, 32))
    VPU_BIST_INTERFACE_B4_DATA = bin_listTobin(VPU_BIST_INTERFACE_B4_DATA)
    VPU_BIST_INTERFACE_B4_DATA = binTohex(VPU_BIST_INTERFACE_B4_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B4_DATA)
    # VPU_BIST_INTERFACE_B4_ADDR
    VPU_BIST_INTERFACE_B4_ADDR = []
    VPU_BIST_INTERFACE_B4_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b4_ans_ctrl = vpu_dict["vpu_BIST_para_o_b4_ans_ctrl"]
    VPU_BIST_INTERFACE_B4_ADDR.append(intToBin(vpu_BIST_para_o_b4_ans_ctrl, 3))
    vpu_BIST_para_o_b4_ans_addr = vpu_dict["vpu_BIST_para_o_b4_ans_addr"]
    VPU_BIST_INTERFACE_B4_ADDR.append(intToBin(vpu_BIST_para_o_b4_ans_addr, 21))
    VPU_BIST_INTERFACE_B4_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B4_ADDR)
    VPU_BIST_INTERFACE_B4_ADDR = binTohex(VPU_BIST_INTERFACE_B4_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B4_ADDR)
    # VPU_BIST_INTERFACE_B5_DATA
    VPU_BIST_INTERFACE_B5_DATA = []
    vpu_BIST_para_o_b5_ans_data = vpu_dict["vpu_BIST_para_o_b5_ans_data"]
    VPU_BIST_INTERFACE_B5_DATA.append(intToBin(vpu_BIST_para_o_b5_ans_data, 32))
    VPU_BIST_INTERFACE_B5_DATA = bin_listTobin(VPU_BIST_INTERFACE_B5_DATA)
    VPU_BIST_INTERFACE_B5_DATA = binTohex(VPU_BIST_INTERFACE_B5_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B5_DATA)
    # VPU_BIST_INTERFACE_B5_ADDR
    VPU_BIST_INTERFACE_B5_ADDR = []
    VPU_BIST_INTERFACE_B5_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b5_ans_ctrl = vpu_dict["vpu_BIST_para_o_b5_ans_ctrl"]
    VPU_BIST_INTERFACE_B5_ADDR.append(intToBin(vpu_BIST_para_o_b5_ans_ctrl, 3))
    vpu_BIST_para_o_b5_ans_addr = vpu_dict["vpu_BIST_para_o_b5_ans_addr"]
    VPU_BIST_INTERFACE_B5_ADDR.append(intToBin(vpu_BIST_para_o_b5_ans_addr, 21))
    VPU_BIST_INTERFACE_B5_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B5_ADDR)
    VPU_BIST_INTERFACE_B5_ADDR = binTohex(VPU_BIST_INTERFACE_B5_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B5_ADDR)
    # VPU_BIST_INTERFACE_B6_DATA
    VPU_BIST_INTERFACE_B6_DATA = []
    vpu_BIST_para_o_b6_ans_data = vpu_dict["vpu_BIST_para_o_b6_ans_data"]
    VPU_BIST_INTERFACE_B6_DATA.append(intToBin(vpu_BIST_para_o_b6_ans_data, 32))
    VPU_BIST_INTERFACE_B6_DATA = bin_listTobin(VPU_BIST_INTERFACE_B6_DATA)
    VPU_BIST_INTERFACE_B6_DATA = binTohex(VPU_BIST_INTERFACE_B6_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B6_DATA)
    # VPU_BIST_INTERFACE_B6_ADDR
    VPU_BIST_INTERFACE_B6_ADDR = []
    VPU_BIST_INTERFACE_B6_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b6_ans_ctrl = vpu_dict["vpu_BIST_para_o_b6_ans_ctrl"]
    VPU_BIST_INTERFACE_B6_ADDR.append(intToBin(vpu_BIST_para_o_b6_ans_ctrl, 3))
    vpu_BIST_para_o_b6_ans_addr = vpu_dict["vpu_BIST_para_o_b6_ans_addr"]
    VPU_BIST_INTERFACE_B6_ADDR.append(intToBin(vpu_BIST_para_o_b6_ans_addr, 21))
    VPU_BIST_INTERFACE_B6_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B6_ADDR)
    VPU_BIST_INTERFACE_B6_ADDR = binTohex(VPU_BIST_INTERFACE_B6_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B6_ADDR)
    # VPU_BIST_INTERFACE_B7_DATA
    VPU_BIST_INTERFACE_B7_DATA = []
    vpu_BIST_para_o_b7_ans_data = vpu_dict["vpu_BIST_para_o_b7_ans_data"]
    VPU_BIST_INTERFACE_B7_DATA.append(intToBin(vpu_BIST_para_o_b7_ans_data, 32))
    VPU_BIST_INTERFACE_B7_DATA = bin_listTobin(VPU_BIST_INTERFACE_B7_DATA)
    VPU_BIST_INTERFACE_B7_DATA = binTohex(VPU_BIST_INTERFACE_B7_DATA, 32)
    register_dict.append(VPU_BIST_INTERFACE_B7_DATA)
    # VPU_BIST_INTERFACE_B7_ADDR
    VPU_BIST_INTERFACE_B7_ADDR = []
    VPU_BIST_INTERFACE_B7_ADDR.append(n_zeros_str(8))
    vpu_BIST_para_o_b7_ans_ctrl = vpu_dict["vpu_BIST_para_o_b7_ans_ctrl"]
    VPU_BIST_INTERFACE_B7_ADDR.append(intToBin(vpu_BIST_para_o_b7_ans_ctrl, 3))
    vpu_BIST_para_o_b7_ans_addr = vpu_dict["vpu_BIST_para_o_b7_ans_addr"]
    VPU_BIST_INTERFACE_B7_ADDR.append(intToBin(vpu_BIST_para_o_b7_ans_addr, 21))
    VPU_BIST_INTERFACE_B7_ADDR = bin_listTobin(VPU_BIST_INTERFACE_B7_ADDR)
    VPU_BIST_INTERFACE_B7_ADDR = binTohex(VPU_BIST_INTERFACE_B7_ADDR, 32)
    register_dict.append(VPU_BIST_INTERFACE_B7_ADDR)
    # CLUS_0_VPU_TOP_SC_AD_0
    CLUS_0_VPU_TOP_SC_AD_0 = []
    CLUS_0_VPU_TOP_SC_AD_0.append(n_zeros_str(4))
    vpu_0_para_clus_0_scw_ad_0 = vpu_dict["vpu_0_para_clus_0_scw_ad_0"]
    CLUS_0_VPU_TOP_SC_AD_0.append(intToBin(vpu_0_para_clus_0_scw_ad_0, 14))
    vpu_0_para_clus_0_scr_ad_0 = vpu_dict["vpu_0_para_clus_0_scr_ad_0"]
    CLUS_0_VPU_TOP_SC_AD_0.append(intToBin(vpu_0_para_clus_0_scr_ad_0, 14))
    CLUS_0_VPU_TOP_SC_AD_0 = bin_listTobin(CLUS_0_VPU_TOP_SC_AD_0)
    CLUS_0_VPU_TOP_SC_AD_0 = binTohex(CLUS_0_VPU_TOP_SC_AD_0, 32)
    register_dict.append(CLUS_0_VPU_TOP_SC_AD_0)
    # CLUS_0_VPU_TOP_AD_0
    CLUS_0_VPU_TOP_AD_0 = []
    CLUS_0_VPU_TOP_AD_0.append(n_zeros_str(18))
    vpu_0_para_clus_0_ad_0 = vpu_dict["vpu_0_para_clus_0_ad_0"]
    CLUS_0_VPU_TOP_AD_0.append(intToBin(vpu_0_para_clus_0_ad_0, 14))
    CLUS_0_VPU_TOP_AD_0 = bin_listTobin(CLUS_0_VPU_TOP_AD_0)
    CLUS_0_VPU_TOP_AD_0 = binTohex(CLUS_0_VPU_TOP_AD_0, 32)
    register_dict.append(CLUS_0_VPU_TOP_AD_0)
    # CLUS_0_VPU_TOP_SC_AD_1
    CLUS_0_VPU_TOP_SC_AD_1 = []
    CLUS_0_VPU_TOP_SC_AD_1.append(n_zeros_str(4))
    vpu_0_para_clus_0_scw_ad_1 = vpu_dict["vpu_0_para_clus_0_scw_ad_1"]
    CLUS_0_VPU_TOP_SC_AD_1.append(intToBin(vpu_0_para_clus_0_scw_ad_1, 14))
    vpu_0_para_clus_0_scr_ad_1 = vpu_dict["vpu_0_para_clus_0_scr_ad_1"]
    CLUS_0_VPU_TOP_SC_AD_1.append(intToBin(vpu_0_para_clus_0_scr_ad_1, 14))
    CLUS_0_VPU_TOP_SC_AD_1 = bin_listTobin(CLUS_0_VPU_TOP_SC_AD_1)
    CLUS_0_VPU_TOP_SC_AD_1 = binTohex(CLUS_0_VPU_TOP_SC_AD_1, 32)
    register_dict.append(CLUS_0_VPU_TOP_SC_AD_1)
    # CLUS_0_VPU_TOP_AD_1
    CLUS_0_VPU_TOP_AD_1 = []
    CLUS_0_VPU_TOP_AD_1.append(n_zeros_str(18))
    vpu_0_para_clus_0_ad_1 = vpu_dict["vpu_0_para_clus_0_ad_1"]
    CLUS_0_VPU_TOP_AD_1.append(intToBin(vpu_0_para_clus_0_ad_1, 14))
    CLUS_0_VPU_TOP_AD_1 = bin_listTobin(CLUS_0_VPU_TOP_AD_1)
    CLUS_0_VPU_TOP_AD_1 = binTohex(CLUS_0_VPU_TOP_AD_1, 32)
    register_dict.append(CLUS_0_VPU_TOP_AD_1)
    # CLUS_0_VPU_TOP_SC_AD_2
    CLUS_0_VPU_TOP_SC_AD_2 = []
    CLUS_0_VPU_TOP_SC_AD_2.append(n_zeros_str(4))
    vpu_0_para_clus_0_scw_ad_2 = vpu_dict["vpu_0_para_clus_0_scw_ad_2"]
    CLUS_0_VPU_TOP_SC_AD_2.append(intToBin(vpu_0_para_clus_0_scw_ad_2, 14))
    vpu_0_para_clus_0_scr_ad_2 = vpu_dict["vpu_0_para_clus_0_scr_ad_2"]
    CLUS_0_VPU_TOP_SC_AD_2.append(intToBin(vpu_0_para_clus_0_scr_ad_2, 14))
    CLUS_0_VPU_TOP_SC_AD_2 = bin_listTobin(CLUS_0_VPU_TOP_SC_AD_2)
    CLUS_0_VPU_TOP_SC_AD_2 = binTohex(CLUS_0_VPU_TOP_SC_AD_2, 32)
    register_dict.append(CLUS_0_VPU_TOP_SC_AD_2)
    # CLUS_0_VPU_TOP_AD_2
    CLUS_0_VPU_TOP_AD_2 = []
    CLUS_0_VPU_TOP_AD_2.append(n_zeros_str(18))
    vpu_0_para_clus_0_ad_2 = vpu_dict["vpu_0_para_clus_0_ad_2"]
    CLUS_0_VPU_TOP_AD_2.append(intToBin(vpu_0_para_clus_0_ad_2, 14))
    CLUS_0_VPU_TOP_AD_2 = bin_listTobin(CLUS_0_VPU_TOP_AD_2)
    CLUS_0_VPU_TOP_AD_2 = binTohex(CLUS_0_VPU_TOP_AD_2, 32)
    register_dict.append(CLUS_0_VPU_TOP_AD_2)
    # CLUS_0_VPU_TOP_SC_AD_3
    CLUS_0_VPU_TOP_SC_AD_3 = []
    CLUS_0_VPU_TOP_SC_AD_3.append(n_zeros_str(4))
    vpu_0_para_clus_0_scw_ad_3 = vpu_dict["vpu_0_para_clus_0_scw_ad_3"]
    CLUS_0_VPU_TOP_SC_AD_3.append(intToBin(vpu_0_para_clus_0_scw_ad_3, 14))
    vpu_0_para_clus_0_scr_ad_3 = vpu_dict["vpu_0_para_clus_0_scr_ad_3"]
    CLUS_0_VPU_TOP_SC_AD_3.append(intToBin(vpu_0_para_clus_0_scr_ad_3, 14))
    CLUS_0_VPU_TOP_SC_AD_3 = bin_listTobin(CLUS_0_VPU_TOP_SC_AD_3)
    CLUS_0_VPU_TOP_SC_AD_3 = binTohex(CLUS_0_VPU_TOP_SC_AD_3, 32)
    register_dict.append(CLUS_0_VPU_TOP_SC_AD_3)
    # CLUS_0_VPU_TOP_AD_3
    CLUS_0_VPU_TOP_AD_3 = []
    CLUS_0_VPU_TOP_AD_3.append(n_zeros_str(18))
    vpu_0_para_clus_0_ad_3 = vpu_dict["vpu_0_para_clus_0_ad_3"]
    CLUS_0_VPU_TOP_AD_3.append(intToBin(vpu_0_para_clus_0_ad_3, 14))
    CLUS_0_VPU_TOP_AD_3 = bin_listTobin(CLUS_0_VPU_TOP_AD_3)
    CLUS_0_VPU_TOP_AD_3 = binTohex(CLUS_0_VPU_TOP_AD_3, 32)
    register_dict.append(CLUS_0_VPU_TOP_AD_3)
    # CLUS_0_VPU_BLOCK_AD_JUMPER_0
    CLUS_0_VPU_BLOCK_AD_JUMPER_0 = []
    CLUS_0_VPU_BLOCK_AD_JUMPER_0.append(n_zeros_str(15))
    vpu_0_para_clus_0_block_ad_jump_0 = vpu_dict["vpu_0_para_clus_0_block_ad_jump_0"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_0_para_clus_0_block_ad_jump_0, 14))
    vpu_0_para_clus_0_block_ad_mode_enable = vpu_dict["vpu_0_para_clus_0_block_ad_mode_enable"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_0_para_clus_0_block_ad_mode_enable, 3))
    CLUS_0_VPU_BLOCK_AD_JUMPER_0 = bin_listTobin(CLUS_0_VPU_BLOCK_AD_JUMPER_0)
    CLUS_0_VPU_BLOCK_AD_JUMPER_0 = binTohex(CLUS_0_VPU_BLOCK_AD_JUMPER_0, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_AD_JUMPER_0)
    # CLUS_0_VPU_BLOCK_AD_JUMPER_1
    CLUS_0_VPU_BLOCK_AD_JUMPER_1 = []
    CLUS_0_VPU_BLOCK_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_0_para_clus_0_block_ad_jump_condit1 = vpu_dict["vpu_0_para_clus_0_block_ad_jump_condit1"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_ad_jump_condit1, 10))
    vpu_0_para_clus_0_block_ad_jump_condit0 = vpu_dict["vpu_0_para_clus_0_block_ad_jump_condit0"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_ad_jump_condit0, 10))
    CLUS_0_VPU_BLOCK_AD_JUMPER_1 = bin_listTobin(CLUS_0_VPU_BLOCK_AD_JUMPER_1)
    CLUS_0_VPU_BLOCK_AD_JUMPER_1 = binTohex(CLUS_0_VPU_BLOCK_AD_JUMPER_1, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_AD_JUMPER_1)
    # CLUS_0_VPU_BLOCK_AD_JUMPER_2
    CLUS_0_VPU_BLOCK_AD_JUMPER_2 = []
    CLUS_0_VPU_BLOCK_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_0_para_clus_0_block_ad_jump_2 = vpu_dict["vpu_0_para_clus_0_block_ad_jump_2"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_ad_jump_2, 14))
    vpu_0_para_clus_0_block_ad_jump_1 = vpu_dict["vpu_0_para_clus_0_block_ad_jump_1"]
    CLUS_0_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_ad_jump_1, 14))
    CLUS_0_VPU_BLOCK_AD_JUMPER_2 = bin_listTobin(CLUS_0_VPU_BLOCK_AD_JUMPER_2)
    CLUS_0_VPU_BLOCK_AD_JUMPER_2 = binTohex(CLUS_0_VPU_BLOCK_AD_JUMPER_2, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_AD_JUMPER_2)
    # CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0 = []
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_0_para_clus_0_block_scr_ad_jump_0 = vpu_dict["vpu_0_para_clus_0_block_scr_ad_jump_0"]
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0.append(intToBin(vpu_0_para_clus_0_block_scr_ad_jump_0, 14))
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0 = bin_listTobin(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0)
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0 = binTohex(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_0)
    # CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1 = []
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_0_para_clus_0_block_scr_ad_jump_condit1 = vpu_dict["vpu_0_para_clus_0_block_scr_ad_jump_condit1"]
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_scr_ad_jump_condit1, 10))
    vpu_0_para_clus_0_block_scr_ad_jump_condit0 = vpu_dict["vpu_0_para_clus_0_block_scr_ad_jump_condit0"]
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_scr_ad_jump_condit0, 10))
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1 = bin_listTobin(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1)
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1 = binTohex(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_1)
    # CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2 = []
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_0_para_clus_0_block_scr_ad_jump_2 = vpu_dict["vpu_0_para_clus_0_block_scr_ad_jump_2"]
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_scr_ad_jump_2, 14))
    vpu_0_para_clus_0_block_scr_ad_jump_1 = vpu_dict["vpu_0_para_clus_0_block_scr_ad_jump_1"]
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_scr_ad_jump_1, 14))
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2 = bin_listTobin(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2)
    CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2 = binTohex(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCR_AD_JUMPER_2)
    # CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0 = []
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_0_para_clus_0_block_scw_ad_jump_0 = vpu_dict["vpu_0_para_clus_0_block_scw_ad_jump_0"]
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0.append(intToBin(vpu_0_para_clus_0_block_scw_ad_jump_0, 14))
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0 = bin_listTobin(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0)
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0 = binTohex(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_0)
    # CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1 = []
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_0_para_clus_0_block_scw_ad_jump_condit1 = vpu_dict["vpu_0_para_clus_0_block_scw_ad_jump_condit1"]
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_scw_ad_jump_condit1, 10))
    vpu_0_para_clus_0_block_scw_ad_jump_condit0 = vpu_dict["vpu_0_para_clus_0_block_scw_ad_jump_condit0"]
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_0_para_clus_0_block_scw_ad_jump_condit0, 10))
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1 = bin_listTobin(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1)
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1 = binTohex(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_1)
    # CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2 = []
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_0_para_clus_0_block_scw_ad_jump_2 = vpu_dict["vpu_0_para_clus_0_block_scw_ad_jump_2"]
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_scw_ad_jump_2, 14))
    vpu_0_para_clus_0_block_scw_ad_jump_1 = vpu_dict["vpu_0_para_clus_0_block_scw_ad_jump_1"]
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_0_para_clus_0_block_scw_ad_jump_1, 14))
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2 = bin_listTobin(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2)
    CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2 = binTohex(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2, 32)
    register_dict.append(CLUS_0_VPU_BLOCK_SCW_AD_JUMPER_2)
    # CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX = []
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(n_zeros_str(13))
    vpu_0_para_clus_0_line_buffer_w_max = vpu_dict["vpu_0_para_clus_0_line_buffer_w_max"]
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_0_para_clus_0_line_buffer_w_max, 9))
    vpu_0_para_clus_0_line_buffer_h_max = vpu_dict["vpu_0_para_clus_0_line_buffer_h_max"]
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_0_para_clus_0_line_buffer_h_max, 10))
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX = bin_listTobin(CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX = binTohex(CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX, 32)
    register_dict.append(CLUS_0_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    # CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH = []
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH.append(n_zeros_str(26))
    vpu_0_para_clus_0_kernal_h = vpu_dict["vpu_0_para_clus_0_kernal_h"]
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_0_para_clus_0_kernal_h, 3))
    vpu_0_para_clus_0_kernal_h_stride = vpu_dict["vpu_0_para_clus_0_kernal_h_stride"]
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_0_para_clus_0_kernal_h_stride, 3))
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH = bin_listTobin(CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH)
    CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH = binTohex(CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH, 32)
    register_dict.append(CLUS_0_VPU_GLOBAL_LINE_BUFFER_KH)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1 = []
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_0_para_clus_0_output_l1_step = vpu_dict["vpu_0_para_clus_0_output_l1_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_output_l1_step, 9))
    vpu_0_para_clus_0_output_l1_condition = vpu_dict["vpu_0_para_clus_0_output_l1_condition"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_output_l1_condition, 9))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1 = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1 = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2 = []
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_0_para_clus_0_output_l2_step = vpu_dict["vpu_0_para_clus_0_output_l2_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_output_l2_step, 9))
    vpu_0_para_clus_0_output_l2_condition = vpu_dict["vpu_0_para_clus_0_output_l2_condition"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_output_l2_condition, 9))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2 = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2 = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3 = []
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_0_para_clus_0_output_l3_step = vpu_dict["vpu_0_para_clus_0_output_l3_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_output_l3_step, 9))
    vpu_0_para_clus_0_output_l3_condition = vpu_dict["vpu_0_para_clus_0_output_l3_condition"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_output_l3_condition, 9))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3 = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3 = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = []
    vpu_0_para_clus_0_output_l1_addr_step = vpu_dict["vpu_0_para_clus_0_output_l1_addr_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_0_para_clus_0_output_l1_addr_step, 32))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = []
    vpu_0_para_clus_0_output_l2_addr_step = vpu_dict["vpu_0_para_clus_0_output_l2_addr_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_0_para_clus_0_output_l2_addr_step, 32))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    # CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = []
    vpu_0_para_clus_0_output_l3_addr_step = vpu_dict["vpu_0_para_clus_0_output_l3_addr_step"]
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_0_para_clus_0_output_l3_addr_step, 32))
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = binTohex(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = []
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_0_para_clus_0_scw_l1_step = vpu_dict["vpu_0_para_clus_0_scw_l1_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_scw_l1_step, 9))
    vpu_0_para_clus_0_scw_l1_condition = vpu_dict["vpu_0_para_clus_0_scw_l1_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_scw_l1_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = []
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_0_para_clus_0_scw_l2_step = vpu_dict["vpu_0_para_clus_0_scw_l2_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_scw_l2_step, 9))
    vpu_0_para_clus_0_scw_l2_condition = vpu_dict["vpu_0_para_clus_0_scw_l2_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_scw_l2_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = []
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_0_para_clus_0_scw_l3_step = vpu_dict["vpu_0_para_clus_0_scw_l3_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_scw_l3_step, 9))
    vpu_0_para_clus_0_scw_l3_condition = vpu_dict["vpu_0_para_clus_0_scw_l3_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_scw_l3_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = []
    vpu_0_para_clus_0_scw_l1_addr_step = vpu_dict["vpu_0_para_clus_0_scw_l1_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_0_para_clus_0_scw_l1_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = []
    vpu_0_para_clus_0_scw_l2_addr_step = vpu_dict["vpu_0_para_clus_0_scw_l2_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_0_para_clus_0_scw_l2_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = []
    vpu_0_para_clus_0_scw_l3_addr_step = vpu_dict["vpu_0_para_clus_0_scw_l3_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_0_para_clus_0_scw_l3_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = []
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_0_para_clus_0_scr_l1_step = vpu_dict["vpu_0_para_clus_0_scr_l1_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_scr_l1_step, 9))
    vpu_0_para_clus_0_scr_l1_condition = vpu_dict["vpu_0_para_clus_0_scr_l1_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_0_para_clus_0_scr_l1_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = []
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_0_para_clus_0_scr_l2_step = vpu_dict["vpu_0_para_clus_0_scr_l2_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_scr_l2_step, 9))
    vpu_0_para_clus_0_scr_l2_condition = vpu_dict["vpu_0_para_clus_0_scr_l2_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_0_para_clus_0_scr_l2_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = []
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_0_para_clus_0_scr_l3_step = vpu_dict["vpu_0_para_clus_0_scr_l3_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_scr_l3_step, 9))
    vpu_0_para_clus_0_scr_l3_condition = vpu_dict["vpu_0_para_clus_0_scr_l3_condition"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_0_para_clus_0_scr_l3_condition, 9))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = []
    vpu_0_para_clus_0_scr_l1_addr_step = vpu_dict["vpu_0_para_clus_0_scr_l1_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_0_para_clus_0_scr_l1_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = []
    vpu_0_para_clus_0_scr_l2_addr_step = vpu_dict["vpu_0_para_clus_0_scr_l2_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_0_para_clus_0_scr_l2_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    # CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = []
    vpu_0_para_clus_0_scr_l3_addr_step = vpu_dict["vpu_0_para_clus_0_scr_l3_addr_step"]
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_0_para_clus_0_scr_l3_addr_step, 32))
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = binTohex(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_0_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    # CLUS_0_RS_WH_RATIO
    CLUS_0_RS_WH_RATIO = []
    vpu_0_para_i_clus_0_resize_param_width_ratio = vpu_dict["vpu_0_para_i_clus_0_resize_param_width_ratio"]
    CLUS_0_RS_WH_RATIO.append(intToBin(vpu_0_para_i_clus_0_resize_param_width_ratio, 16))
    vpu_0_para_i_clus_0_resize_param_height_ratio = vpu_dict["vpu_0_para_i_clus_0_resize_param_height_ratio"]
    CLUS_0_RS_WH_RATIO.append(intToBin(vpu_0_para_i_clus_0_resize_param_height_ratio, 16))
    CLUS_0_RS_WH_RATIO = bin_listTobin(CLUS_0_RS_WH_RATIO)
    CLUS_0_RS_WH_RATIO = binTohex(CLUS_0_RS_WH_RATIO, 32)
    register_dict.append(CLUS_0_RS_WH_RATIO)
    # CLUS_0_RS_INPUT_WH
    CLUS_0_RS_INPUT_WH = []
    CLUS_0_RS_INPUT_WH.append(n_zeros_str(12))
    vpu_0_para_i_clus_0_resize_param_input_width = vpu_dict["vpu_0_para_i_clus_0_resize_param_input_width"]
    CLUS_0_RS_INPUT_WH.append(intToBin(vpu_0_para_i_clus_0_resize_param_input_width, 9))
    vpu_0_para_i_clus_0_resize_param_input_height = vpu_dict["vpu_0_para_i_clus_0_resize_param_input_height"]
    CLUS_0_RS_INPUT_WH.append(intToBin(vpu_0_para_i_clus_0_resize_param_input_height, 11))
    CLUS_0_RS_INPUT_WH = bin_listTobin(CLUS_0_RS_INPUT_WH)
    CLUS_0_RS_INPUT_WH = binTohex(CLUS_0_RS_INPUT_WH, 32)
    register_dict.append(CLUS_0_RS_INPUT_WH)
    # CLUS_0_RS_OUTPUT_WH
    CLUS_0_RS_OUTPUT_WH = []
    CLUS_0_RS_OUTPUT_WH.append(n_zeros_str(10))
    vpu_0_para_i_clus_0_resize_param_output_width = vpu_dict["vpu_0_para_i_clus_0_resize_param_output_width"]
    CLUS_0_RS_OUTPUT_WH.append(intToBin(vpu_0_para_i_clus_0_resize_param_output_width, 11))
    vpu_0_para_i_clus_0_resize_param_output_height = vpu_dict["vpu_0_para_i_clus_0_resize_param_output_height"]
    CLUS_0_RS_OUTPUT_WH.append(intToBin(vpu_0_para_i_clus_0_resize_param_output_height, 11))
    CLUS_0_RS_OUTPUT_WH = bin_listTobin(CLUS_0_RS_OUTPUT_WH)
    CLUS_0_RS_OUTPUT_WH = binTohex(CLUS_0_RS_OUTPUT_WH, 32)
    register_dict.append(CLUS_0_RS_OUTPUT_WH)
    # CLUS_0_PL_INPUT_WH
    CLUS_0_PL_INPUT_WH = []
    CLUS_0_PL_INPUT_WH.append(n_zeros_str(12))
    vpu_0_para_i_clus_0_pooling_param_input_width = vpu_dict["vpu_0_para_i_clus_0_pooling_param_input_width"]
    CLUS_0_PL_INPUT_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_param_input_width, 9))
    vpu_0_para_i_clus_0_pooling_param_input_height = vpu_dict["vpu_0_para_i_clus_0_pooling_param_input_height"]
    CLUS_0_PL_INPUT_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_param_input_height, 11))
    CLUS_0_PL_INPUT_WH = bin_listTobin(CLUS_0_PL_INPUT_WH)
    CLUS_0_PL_INPUT_WH = binTohex(CLUS_0_PL_INPUT_WH, 32)
    register_dict.append(CLUS_0_PL_INPUT_WH)
    # CLUS_0_PL_OUTPUT_WH
    CLUS_0_PL_OUTPUT_WH = []
    CLUS_0_PL_OUTPUT_WH.append(n_zeros_str(10))
    vpu_0_para_i_clus_0_pooling_param_output_width = vpu_dict["vpu_0_para_i_clus_0_pooling_param_output_width"]
    CLUS_0_PL_OUTPUT_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_param_output_width, 11))
    vpu_0_para_i_clus_0_pooling_param_output_height = vpu_dict["vpu_0_para_i_clus_0_pooling_param_output_height"]
    CLUS_0_PL_OUTPUT_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_param_output_height, 11))
    CLUS_0_PL_OUTPUT_WH = bin_listTobin(CLUS_0_PL_OUTPUT_WH)
    CLUS_0_PL_OUTPUT_WH = binTohex(CLUS_0_PL_OUTPUT_WH, 32)
    register_dict.append(CLUS_0_PL_OUTPUT_WH)
    # CLUS_0_PL_PADDING_WH
    CLUS_0_PL_PADDING_WH = []
    CLUS_0_PL_PADDING_WH.append(n_zeros_str(6))
    vpu_0_para_i_clus_0_pooling_padding_mode = vpu_dict["vpu_0_para_i_clus_0_pooling_padding_mode"]
    CLUS_0_PL_PADDING_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_padding_mode, 4))
    vpu_0_para_i_clus_0_pooling_padding_width = vpu_dict["vpu_0_para_i_clus_0_pooling_padding_width"]
    CLUS_0_PL_PADDING_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_padding_width, 11))
    vpu_0_para_i_clus_0_pooling_padding_height = vpu_dict["vpu_0_para_i_clus_0_pooling_padding_height"]
    CLUS_0_PL_PADDING_WH.append(intToBin(vpu_0_para_i_clus_0_pooling_padding_height, 11))
    CLUS_0_PL_PADDING_WH = bin_listTobin(CLUS_0_PL_PADDING_WH)
    CLUS_0_PL_PADDING_WH = binTohex(CLUS_0_PL_PADDING_WH, 32)
    register_dict.append(CLUS_0_PL_PADDING_WH)
    # CLUS_1_VPU_TOP_SC_AD_0
    CLUS_1_VPU_TOP_SC_AD_0 = []
    CLUS_1_VPU_TOP_SC_AD_0.append(n_zeros_str(4))
    vpu_1_para_clus_1_scw_ad_0 = vpu_dict["vpu_1_para_clus_1_scw_ad_0"]
    CLUS_1_VPU_TOP_SC_AD_0.append(intToBin(vpu_1_para_clus_1_scw_ad_0, 14))
    vpu_1_para_clus_1_scr_ad_0 = vpu_dict["vpu_1_para_clus_1_scr_ad_0"]
    CLUS_1_VPU_TOP_SC_AD_0.append(intToBin(vpu_1_para_clus_1_scr_ad_0, 14))
    CLUS_1_VPU_TOP_SC_AD_0 = bin_listTobin(CLUS_1_VPU_TOP_SC_AD_0)
    CLUS_1_VPU_TOP_SC_AD_0 = binTohex(CLUS_1_VPU_TOP_SC_AD_0, 32)
    register_dict.append(CLUS_1_VPU_TOP_SC_AD_0)
    # CLUS_1_VPU_TOP_AD_0
    CLUS_1_VPU_TOP_AD_0 = []
    CLUS_1_VPU_TOP_AD_0.append(n_zeros_str(18))
    vpu_1_para_clus_1_ad_0 = vpu_dict["vpu_1_para_clus_1_ad_0"]
    CLUS_1_VPU_TOP_AD_0.append(intToBin(vpu_1_para_clus_1_ad_0, 14))
    CLUS_1_VPU_TOP_AD_0 = bin_listTobin(CLUS_1_VPU_TOP_AD_0)
    CLUS_1_VPU_TOP_AD_0 = binTohex(CLUS_1_VPU_TOP_AD_0, 32)
    register_dict.append(CLUS_1_VPU_TOP_AD_0)
    # CLUS_1_VPU_TOP_SC_AD_1
    CLUS_1_VPU_TOP_SC_AD_1 = []
    CLUS_1_VPU_TOP_SC_AD_1.append(n_zeros_str(4))
    vpu_1_para_clus_1_scw_ad_1 = vpu_dict["vpu_1_para_clus_1_scw_ad_1"]
    CLUS_1_VPU_TOP_SC_AD_1.append(intToBin(vpu_1_para_clus_1_scw_ad_1, 14))
    vpu_1_para_clus_1_scr_ad_1 = vpu_dict["vpu_1_para_clus_1_scr_ad_1"]
    CLUS_1_VPU_TOP_SC_AD_1.append(intToBin(vpu_1_para_clus_1_scr_ad_1, 14))
    CLUS_1_VPU_TOP_SC_AD_1 = bin_listTobin(CLUS_1_VPU_TOP_SC_AD_1)
    CLUS_1_VPU_TOP_SC_AD_1 = binTohex(CLUS_1_VPU_TOP_SC_AD_1, 32)
    register_dict.append(CLUS_1_VPU_TOP_SC_AD_1)
    # CLUS_1_VPU_TOP_AD_1
    CLUS_1_VPU_TOP_AD_1 = []
    CLUS_1_VPU_TOP_AD_1.append(n_zeros_str(18))
    vpu_1_para_clus_1_ad_1 = vpu_dict["vpu_1_para_clus_1_ad_1"]
    CLUS_1_VPU_TOP_AD_1.append(intToBin(vpu_1_para_clus_1_ad_1, 14))
    CLUS_1_VPU_TOP_AD_1 = bin_listTobin(CLUS_1_VPU_TOP_AD_1)
    CLUS_1_VPU_TOP_AD_1 = binTohex(CLUS_1_VPU_TOP_AD_1, 32)
    register_dict.append(CLUS_1_VPU_TOP_AD_1)
    # CLUS_1_VPU_TOP_SC_AD_2
    CLUS_1_VPU_TOP_SC_AD_2 = []
    CLUS_1_VPU_TOP_SC_AD_2.append(n_zeros_str(4))
    vpu_1_para_clus_1_scw_ad_2 = vpu_dict["vpu_1_para_clus_1_scw_ad_2"]
    CLUS_1_VPU_TOP_SC_AD_2.append(intToBin(vpu_1_para_clus_1_scw_ad_2, 14))
    vpu_1_para_clus_1_scr_ad_2 = vpu_dict["vpu_1_para_clus_1_scr_ad_2"]
    CLUS_1_VPU_TOP_SC_AD_2.append(intToBin(vpu_1_para_clus_1_scr_ad_2, 14))
    CLUS_1_VPU_TOP_SC_AD_2 = bin_listTobin(CLUS_1_VPU_TOP_SC_AD_2)
    CLUS_1_VPU_TOP_SC_AD_2 = binTohex(CLUS_1_VPU_TOP_SC_AD_2, 32)
    register_dict.append(CLUS_1_VPU_TOP_SC_AD_2)
    # CLUS_1_VPU_TOP_AD_2
    CLUS_1_VPU_TOP_AD_2 = []
    CLUS_1_VPU_TOP_AD_2.append(n_zeros_str(18))
    vpu_1_para_clus_1_ad_2 = vpu_dict["vpu_1_para_clus_1_ad_2"]
    CLUS_1_VPU_TOP_AD_2.append(intToBin(vpu_1_para_clus_1_ad_2, 14))
    CLUS_1_VPU_TOP_AD_2 = bin_listTobin(CLUS_1_VPU_TOP_AD_2)
    CLUS_1_VPU_TOP_AD_2 = binTohex(CLUS_1_VPU_TOP_AD_2, 32)
    register_dict.append(CLUS_1_VPU_TOP_AD_2)
    # CLUS_1_VPU_TOP_SC_AD_3
    CLUS_1_VPU_TOP_SC_AD_3 = []
    CLUS_1_VPU_TOP_SC_AD_3.append(n_zeros_str(4))
    vpu_1_para_clus_1_scw_ad_3 = vpu_dict["vpu_1_para_clus_1_scw_ad_3"]
    CLUS_1_VPU_TOP_SC_AD_3.append(intToBin(vpu_1_para_clus_1_scw_ad_3, 14))
    vpu_1_para_clus_1_scr_ad_3 = vpu_dict["vpu_1_para_clus_1_scr_ad_3"]
    CLUS_1_VPU_TOP_SC_AD_3.append(intToBin(vpu_1_para_clus_1_scr_ad_3, 14))
    CLUS_1_VPU_TOP_SC_AD_3 = bin_listTobin(CLUS_1_VPU_TOP_SC_AD_3)
    CLUS_1_VPU_TOP_SC_AD_3 = binTohex(CLUS_1_VPU_TOP_SC_AD_3, 32)
    register_dict.append(CLUS_1_VPU_TOP_SC_AD_3)
    # CLUS_1_VPU_TOP_AD_3
    CLUS_1_VPU_TOP_AD_3 = []
    CLUS_1_VPU_TOP_AD_3.append(n_zeros_str(18))
    vpu_1_para_clus_1_ad_3 = vpu_dict["vpu_1_para_clus_1_ad_3"]
    CLUS_1_VPU_TOP_AD_3.append(intToBin(vpu_1_para_clus_1_ad_3, 14))
    CLUS_1_VPU_TOP_AD_3 = bin_listTobin(CLUS_1_VPU_TOP_AD_3)
    CLUS_1_VPU_TOP_AD_3 = binTohex(CLUS_1_VPU_TOP_AD_3, 32)
    register_dict.append(CLUS_1_VPU_TOP_AD_3)
    # CLUS_1_VPU_BLOCK_AD_JUMPER_0
    CLUS_1_VPU_BLOCK_AD_JUMPER_0 = []
    CLUS_1_VPU_BLOCK_AD_JUMPER_0.append(n_zeros_str(15))
    vpu_1_para_clus_1_block_ad_jump_0 = vpu_dict["vpu_1_para_clus_1_block_ad_jump_0"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_1_para_clus_1_block_ad_jump_0, 14))
    vpu_1_para_clus_1_block_ad_mode_enable = vpu_dict["vpu_1_para_clus_1_block_ad_mode_enable"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_1_para_clus_1_block_ad_mode_enable, 3))
    CLUS_1_VPU_BLOCK_AD_JUMPER_0 = bin_listTobin(CLUS_1_VPU_BLOCK_AD_JUMPER_0)
    CLUS_1_VPU_BLOCK_AD_JUMPER_0 = binTohex(CLUS_1_VPU_BLOCK_AD_JUMPER_0, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_AD_JUMPER_0)
    # CLUS_1_VPU_BLOCK_AD_JUMPER_1
    CLUS_1_VPU_BLOCK_AD_JUMPER_1 = []
    CLUS_1_VPU_BLOCK_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_1_para_clus_1_block_ad_jump_condit1 = vpu_dict["vpu_1_para_clus_1_block_ad_jump_condit1"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_ad_jump_condit1, 10))
    vpu_1_para_clus_1_block_ad_jump_condit0 = vpu_dict["vpu_1_para_clus_1_block_ad_jump_condit0"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_ad_jump_condit0, 10))
    CLUS_1_VPU_BLOCK_AD_JUMPER_1 = bin_listTobin(CLUS_1_VPU_BLOCK_AD_JUMPER_1)
    CLUS_1_VPU_BLOCK_AD_JUMPER_1 = binTohex(CLUS_1_VPU_BLOCK_AD_JUMPER_1, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_AD_JUMPER_1)
    # CLUS_1_VPU_BLOCK_AD_JUMPER_2
    CLUS_1_VPU_BLOCK_AD_JUMPER_2 = []
    CLUS_1_VPU_BLOCK_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_1_para_clus_1_block_ad_jump_2 = vpu_dict["vpu_1_para_clus_1_block_ad_jump_2"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_ad_jump_2, 14))
    vpu_1_para_clus_1_block_ad_jump_1 = vpu_dict["vpu_1_para_clus_1_block_ad_jump_1"]
    CLUS_1_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_ad_jump_1, 14))
    CLUS_1_VPU_BLOCK_AD_JUMPER_2 = bin_listTobin(CLUS_1_VPU_BLOCK_AD_JUMPER_2)
    CLUS_1_VPU_BLOCK_AD_JUMPER_2 = binTohex(CLUS_1_VPU_BLOCK_AD_JUMPER_2, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_AD_JUMPER_2)
    # CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0 = []
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_1_para_clus_1_block_scr_ad_jump_0 = vpu_dict["vpu_1_para_clus_1_block_scr_ad_jump_0"]
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0.append(intToBin(vpu_1_para_clus_1_block_scr_ad_jump_0, 14))
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0 = bin_listTobin(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0)
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0 = binTohex(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_0)
    # CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1 = []
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_1_para_clus_1_block_scr_ad_jump_condit1 = vpu_dict["vpu_1_para_clus_1_block_scr_ad_jump_condit1"]
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_scr_ad_jump_condit1, 10))
    vpu_1_para_clus_1_block_scr_ad_jump_condit0 = vpu_dict["vpu_1_para_clus_1_block_scr_ad_jump_condit0"]
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_scr_ad_jump_condit0, 10))
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1 = bin_listTobin(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1)
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1 = binTohex(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_1)
    # CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2 = []
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_1_para_clus_1_block_scr_ad_jump_2 = vpu_dict["vpu_1_para_clus_1_block_scr_ad_jump_2"]
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_scr_ad_jump_2, 14))
    vpu_1_para_clus_1_block_scr_ad_jump_1 = vpu_dict["vpu_1_para_clus_1_block_scr_ad_jump_1"]
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_scr_ad_jump_1, 14))
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2 = bin_listTobin(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2)
    CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2 = binTohex(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCR_AD_JUMPER_2)
    # CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0 = []
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_1_para_clus_1_block_scw_ad_jump_0 = vpu_dict["vpu_1_para_clus_1_block_scw_ad_jump_0"]
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0.append(intToBin(vpu_1_para_clus_1_block_scw_ad_jump_0, 14))
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0 = bin_listTobin(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0)
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0 = binTohex(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_0)
    # CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1 = []
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_1_para_clus_1_block_scw_ad_jump_condit1 = vpu_dict["vpu_1_para_clus_1_block_scw_ad_jump_condit1"]
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_scw_ad_jump_condit1, 10))
    vpu_1_para_clus_1_block_scw_ad_jump_condit0 = vpu_dict["vpu_1_para_clus_1_block_scw_ad_jump_condit0"]
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_1_para_clus_1_block_scw_ad_jump_condit0, 10))
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1 = bin_listTobin(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1)
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1 = binTohex(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_1)
    # CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2 = []
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_1_para_clus_1_block_scw_ad_jump_2 = vpu_dict["vpu_1_para_clus_1_block_scw_ad_jump_2"]
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_scw_ad_jump_2, 14))
    vpu_1_para_clus_1_block_scw_ad_jump_1 = vpu_dict["vpu_1_para_clus_1_block_scw_ad_jump_1"]
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_1_para_clus_1_block_scw_ad_jump_1, 14))
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2 = bin_listTobin(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2)
    CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2 = binTohex(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2, 32)
    register_dict.append(CLUS_1_VPU_BLOCK_SCW_AD_JUMPER_2)
    # CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX = []
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(n_zeros_str(13))
    vpu_1_para_clus_1_line_buffer_w_max = vpu_dict["vpu_1_para_clus_1_line_buffer_w_max"]
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_1_para_clus_1_line_buffer_w_max, 9))
    vpu_1_para_clus_1_line_buffer_h_max = vpu_dict["vpu_1_para_clus_1_line_buffer_h_max"]
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_1_para_clus_1_line_buffer_h_max, 10))
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX = bin_listTobin(CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX = binTohex(CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX, 32)
    register_dict.append(CLUS_1_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    # CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH = []
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH.append(n_zeros_str(26))
    vpu_1_para_clus_1_kernal_h = vpu_dict["vpu_1_para_clus_1_kernal_h"]
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_1_para_clus_1_kernal_h, 3))
    vpu_1_para_clus_1_kernal_h_stride = vpu_dict["vpu_1_para_clus_1_kernal_h_stride"]
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_1_para_clus_1_kernal_h_stride, 3))
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH = bin_listTobin(CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH)
    CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH = binTohex(CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH, 32)
    register_dict.append(CLUS_1_VPU_GLOBAL_LINE_BUFFER_KH)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1 = []
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_1_para_clus_1_output_l1_step = vpu_dict["vpu_1_para_clus_1_output_l1_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_output_l1_step, 9))
    vpu_1_para_clus_1_output_l1_condition = vpu_dict["vpu_1_para_clus_1_output_l1_condition"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_output_l1_condition, 9))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1 = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1 = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2 = []
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_1_para_clus_1_output_l2_step = vpu_dict["vpu_1_para_clus_1_output_l2_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_output_l2_step, 9))
    vpu_1_para_clus_1_output_l2_condition = vpu_dict["vpu_1_para_clus_1_output_l2_condition"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_output_l2_condition, 9))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2 = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2 = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3 = []
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_1_para_clus_1_output_l3_step = vpu_dict["vpu_1_para_clus_1_output_l3_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_output_l3_step, 9))
    vpu_1_para_clus_1_output_l3_condition = vpu_dict["vpu_1_para_clus_1_output_l3_condition"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_output_l3_condition, 9))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3 = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3 = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = []
    vpu_1_para_clus_1_output_l1_addr_step = vpu_dict["vpu_1_para_clus_1_output_l1_addr_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_1_para_clus_1_output_l1_addr_step, 32))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = []
    vpu_1_para_clus_1_output_l2_addr_step = vpu_dict["vpu_1_para_clus_1_output_l2_addr_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_1_para_clus_1_output_l2_addr_step, 32))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    # CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = []
    vpu_1_para_clus_1_output_l3_addr_step = vpu_dict["vpu_1_para_clus_1_output_l3_addr_step"]
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_1_para_clus_1_output_l3_addr_step, 32))
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = binTohex(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = []
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_1_para_clus_1_scw_l1_step = vpu_dict["vpu_1_para_clus_1_scw_l1_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_scw_l1_step, 9))
    vpu_1_para_clus_1_scw_l1_condition = vpu_dict["vpu_1_para_clus_1_scw_l1_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_scw_l1_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = []
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_1_para_clus_1_scw_l2_step = vpu_dict["vpu_1_para_clus_1_scw_l2_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_scw_l2_step, 9))
    vpu_1_para_clus_1_scw_l2_condition = vpu_dict["vpu_1_para_clus_1_scw_l2_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_scw_l2_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = []
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_1_para_clus_1_scw_l3_step = vpu_dict["vpu_1_para_clus_1_scw_l3_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_scw_l3_step, 9))
    vpu_1_para_clus_1_scw_l3_condition = vpu_dict["vpu_1_para_clus_1_scw_l3_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_scw_l3_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = []
    vpu_1_para_clus_1_scw_l1_addr_step = vpu_dict["vpu_1_para_clus_1_scw_l1_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_1_para_clus_1_scw_l1_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = []
    vpu_1_para_clus_1_scw_l2_addr_step = vpu_dict["vpu_1_para_clus_1_scw_l2_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_1_para_clus_1_scw_l2_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = []
    vpu_1_para_clus_1_scw_l3_addr_step = vpu_dict["vpu_1_para_clus_1_scw_l3_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_1_para_clus_1_scw_l3_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = []
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_1_para_clus_1_scr_l1_step = vpu_dict["vpu_1_para_clus_1_scr_l1_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_scr_l1_step, 9))
    vpu_1_para_clus_1_scr_l1_condition = vpu_dict["vpu_1_para_clus_1_scr_l1_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_1_para_clus_1_scr_l1_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = []
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_1_para_clus_1_scr_l2_step = vpu_dict["vpu_1_para_clus_1_scr_l2_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_scr_l2_step, 9))
    vpu_1_para_clus_1_scr_l2_condition = vpu_dict["vpu_1_para_clus_1_scr_l2_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_1_para_clus_1_scr_l2_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = []
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_1_para_clus_1_scr_l3_step = vpu_dict["vpu_1_para_clus_1_scr_l3_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_scr_l3_step, 9))
    vpu_1_para_clus_1_scr_l3_condition = vpu_dict["vpu_1_para_clus_1_scr_l3_condition"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_1_para_clus_1_scr_l3_condition, 9))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = []
    vpu_1_para_clus_1_scr_l1_addr_step = vpu_dict["vpu_1_para_clus_1_scr_l1_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_1_para_clus_1_scr_l1_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = []
    vpu_1_para_clus_1_scr_l2_addr_step = vpu_dict["vpu_1_para_clus_1_scr_l2_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_1_para_clus_1_scr_l2_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    # CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = []
    vpu_1_para_clus_1_scr_l3_addr_step = vpu_dict["vpu_1_para_clus_1_scr_l3_addr_step"]
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_1_para_clus_1_scr_l3_addr_step, 32))
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = binTohex(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_1_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    # CLUS_1_RS_WH_RATIO
    CLUS_1_RS_WH_RATIO = []
    vpu_1_para_i_clus_1_resize_param_width_ratio = vpu_dict["vpu_1_para_i_clus_1_resize_param_width_ratio"]
    CLUS_1_RS_WH_RATIO.append(intToBin(vpu_1_para_i_clus_1_resize_param_width_ratio, 16))
    vpu_1_para_i_clus_1_resize_param_height_ratio = vpu_dict["vpu_1_para_i_clus_1_resize_param_height_ratio"]
    CLUS_1_RS_WH_RATIO.append(intToBin(vpu_1_para_i_clus_1_resize_param_height_ratio, 16))
    CLUS_1_RS_WH_RATIO = bin_listTobin(CLUS_1_RS_WH_RATIO)
    CLUS_1_RS_WH_RATIO = binTohex(CLUS_1_RS_WH_RATIO, 32)
    register_dict.append(CLUS_1_RS_WH_RATIO)
    # CLUS_1_RS_INPUT_WH
    CLUS_1_RS_INPUT_WH = []
    CLUS_1_RS_INPUT_WH.append(n_zeros_str(12))
    vpu_1_para_i_clus_1_resize_param_input_width = vpu_dict["vpu_1_para_i_clus_1_resize_param_input_width"]
    CLUS_1_RS_INPUT_WH.append(intToBin(vpu_1_para_i_clus_1_resize_param_input_width, 9))
    vpu_1_para_i_clus_1_resize_param_input_height = vpu_dict["vpu_1_para_i_clus_1_resize_param_input_height"]
    CLUS_1_RS_INPUT_WH.append(intToBin(vpu_1_para_i_clus_1_resize_param_input_height, 11))
    CLUS_1_RS_INPUT_WH = bin_listTobin(CLUS_1_RS_INPUT_WH)
    CLUS_1_RS_INPUT_WH = binTohex(CLUS_1_RS_INPUT_WH, 32)
    register_dict.append(CLUS_1_RS_INPUT_WH)
    # CLUS_1_RS_OUTPUT_WH
    CLUS_1_RS_OUTPUT_WH = []
    CLUS_1_RS_OUTPUT_WH.append(n_zeros_str(10))
    vpu_1_para_i_clus_1_resize_param_output_width = vpu_dict["vpu_1_para_i_clus_1_resize_param_output_width"]
    CLUS_1_RS_OUTPUT_WH.append(intToBin(vpu_1_para_i_clus_1_resize_param_output_width, 11))
    vpu_1_para_i_clus_1_resize_param_output_height = vpu_dict["vpu_1_para_i_clus_1_resize_param_output_height"]
    CLUS_1_RS_OUTPUT_WH.append(intToBin(vpu_1_para_i_clus_1_resize_param_output_height, 11))
    CLUS_1_RS_OUTPUT_WH = bin_listTobin(CLUS_1_RS_OUTPUT_WH)
    CLUS_1_RS_OUTPUT_WH = binTohex(CLUS_1_RS_OUTPUT_WH, 32)
    register_dict.append(CLUS_1_RS_OUTPUT_WH)
    # CLUS_1_PL_INPUT_WH
    CLUS_1_PL_INPUT_WH = []
    CLUS_1_PL_INPUT_WH.append(n_zeros_str(12))
    vpu_1_para_i_clus_1_pooling_param_input_width = vpu_dict["vpu_1_para_i_clus_1_pooling_param_input_width"]
    CLUS_1_PL_INPUT_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_param_input_width, 9))
    vpu_1_para_i_clus_1_pooling_param_input_height = vpu_dict["vpu_1_para_i_clus_1_pooling_param_input_height"]
    CLUS_1_PL_INPUT_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_param_input_height, 11))
    CLUS_1_PL_INPUT_WH = bin_listTobin(CLUS_1_PL_INPUT_WH)
    CLUS_1_PL_INPUT_WH = binTohex(CLUS_1_PL_INPUT_WH, 32)
    register_dict.append(CLUS_1_PL_INPUT_WH)
    # CLUS_1_PL_OUTPUT_WH
    CLUS_1_PL_OUTPUT_WH = []
    CLUS_1_PL_OUTPUT_WH.append(n_zeros_str(10))
    vpu_1_para_i_clus_1_pooling_param_output_width = vpu_dict["vpu_1_para_i_clus_1_pooling_param_output_width"]
    CLUS_1_PL_OUTPUT_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_param_output_width, 11))
    vpu_1_para_i_clus_1_pooling_param_output_height = vpu_dict["vpu_1_para_i_clus_1_pooling_param_output_height"]
    CLUS_1_PL_OUTPUT_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_param_output_height, 11))
    CLUS_1_PL_OUTPUT_WH = bin_listTobin(CLUS_1_PL_OUTPUT_WH)
    CLUS_1_PL_OUTPUT_WH = binTohex(CLUS_1_PL_OUTPUT_WH, 32)
    register_dict.append(CLUS_1_PL_OUTPUT_WH)
    # CLUS_1_PL_PADDING_WH
    CLUS_1_PL_PADDING_WH = []
    CLUS_1_PL_PADDING_WH.append(n_zeros_str(6))
    vpu_1_para_i_clus_1_pooling_padding_mode = vpu_dict["vpu_1_para_i_clus_1_pooling_padding_mode"]
    CLUS_1_PL_PADDING_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_padding_mode, 4))
    vpu_1_para_i_clus_1_pooling_padding_width = vpu_dict["vpu_1_para_i_clus_1_pooling_padding_width"]
    CLUS_1_PL_PADDING_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_padding_width, 11))
    vpu_1_para_i_clus_1_pooling_padding_height = vpu_dict["vpu_1_para_i_clus_1_pooling_padding_height"]
    CLUS_1_PL_PADDING_WH.append(intToBin(vpu_1_para_i_clus_1_pooling_padding_height, 11))
    CLUS_1_PL_PADDING_WH = bin_listTobin(CLUS_1_PL_PADDING_WH)
    CLUS_1_PL_PADDING_WH = binTohex(CLUS_1_PL_PADDING_WH, 32)
    register_dict.append(CLUS_1_PL_PADDING_WH)
    # CLUS_2_VPU_TOP_SC_AD_0
    CLUS_2_VPU_TOP_SC_AD_0 = []
    CLUS_2_VPU_TOP_SC_AD_0.append(n_zeros_str(4))
    vpu_2_para_clus_2_scw_ad_0 = vpu_dict["vpu_2_para_clus_2_scw_ad_0"]
    CLUS_2_VPU_TOP_SC_AD_0.append(intToBin(vpu_2_para_clus_2_scw_ad_0, 14))
    vpu_2_para_clus_2_scr_ad_0 = vpu_dict["vpu_2_para_clus_2_scr_ad_0"]
    CLUS_2_VPU_TOP_SC_AD_0.append(intToBin(vpu_2_para_clus_2_scr_ad_0, 14))
    CLUS_2_VPU_TOP_SC_AD_0 = bin_listTobin(CLUS_2_VPU_TOP_SC_AD_0)
    CLUS_2_VPU_TOP_SC_AD_0 = binTohex(CLUS_2_VPU_TOP_SC_AD_0, 32)
    register_dict.append(CLUS_2_VPU_TOP_SC_AD_0)
    # CLUS_2_VPU_TOP_AD_0
    CLUS_2_VPU_TOP_AD_0 = []
    CLUS_2_VPU_TOP_AD_0.append(n_zeros_str(18))
    vpu_2_para_clus_2_ad_0 = vpu_dict["vpu_2_para_clus_2_ad_0"]
    CLUS_2_VPU_TOP_AD_0.append(intToBin(vpu_2_para_clus_2_ad_0, 14))
    CLUS_2_VPU_TOP_AD_0 = bin_listTobin(CLUS_2_VPU_TOP_AD_0)
    CLUS_2_VPU_TOP_AD_0 = binTohex(CLUS_2_VPU_TOP_AD_0, 32)
    register_dict.append(CLUS_2_VPU_TOP_AD_0)
    # CLUS_2_VPU_TOP_SC_AD_1
    CLUS_2_VPU_TOP_SC_AD_1 = []
    CLUS_2_VPU_TOP_SC_AD_1.append(n_zeros_str(4))
    vpu_2_para_clus_2_scw_ad_1 = vpu_dict["vpu_2_para_clus_2_scw_ad_1"]
    CLUS_2_VPU_TOP_SC_AD_1.append(intToBin(vpu_2_para_clus_2_scw_ad_1, 14))
    vpu_2_para_clus_2_scr_ad_1 = vpu_dict["vpu_2_para_clus_2_scr_ad_1"]
    CLUS_2_VPU_TOP_SC_AD_1.append(intToBin(vpu_2_para_clus_2_scr_ad_1, 14))
    CLUS_2_VPU_TOP_SC_AD_1 = bin_listTobin(CLUS_2_VPU_TOP_SC_AD_1)
    CLUS_2_VPU_TOP_SC_AD_1 = binTohex(CLUS_2_VPU_TOP_SC_AD_1, 32)
    register_dict.append(CLUS_2_VPU_TOP_SC_AD_1)
    # CLUS_2_VPU_TOP_AD_1
    CLUS_2_VPU_TOP_AD_1 = []
    CLUS_2_VPU_TOP_AD_1.append(n_zeros_str(18))
    vpu_2_para_clus_2_ad_1 = vpu_dict["vpu_2_para_clus_2_ad_1"]
    CLUS_2_VPU_TOP_AD_1.append(intToBin(vpu_2_para_clus_2_ad_1, 14))
    CLUS_2_VPU_TOP_AD_1 = bin_listTobin(CLUS_2_VPU_TOP_AD_1)
    CLUS_2_VPU_TOP_AD_1 = binTohex(CLUS_2_VPU_TOP_AD_1, 32)
    register_dict.append(CLUS_2_VPU_TOP_AD_1)
    # CLUS_2_VPU_TOP_SC_AD_2
    CLUS_2_VPU_TOP_SC_AD_2 = []
    CLUS_2_VPU_TOP_SC_AD_2.append(n_zeros_str(4))
    vpu_2_para_clus_2_scw_ad_2 = vpu_dict["vpu_2_para_clus_2_scw_ad_2"]
    CLUS_2_VPU_TOP_SC_AD_2.append(intToBin(vpu_2_para_clus_2_scw_ad_2, 14))
    vpu_2_para_clus_2_scr_ad_2 = vpu_dict["vpu_2_para_clus_2_scr_ad_2"]
    CLUS_2_VPU_TOP_SC_AD_2.append(intToBin(vpu_2_para_clus_2_scr_ad_2, 14))
    CLUS_2_VPU_TOP_SC_AD_2 = bin_listTobin(CLUS_2_VPU_TOP_SC_AD_2)
    CLUS_2_VPU_TOP_SC_AD_2 = binTohex(CLUS_2_VPU_TOP_SC_AD_2, 32)
    register_dict.append(CLUS_2_VPU_TOP_SC_AD_2)
    # CLUS_2_VPU_TOP_AD_2
    CLUS_2_VPU_TOP_AD_2 = []
    CLUS_2_VPU_TOP_AD_2.append(n_zeros_str(18))
    vpu_2_para_clus_2_ad_2 = vpu_dict["vpu_2_para_clus_2_ad_2"]
    CLUS_2_VPU_TOP_AD_2.append(intToBin(vpu_2_para_clus_2_ad_2, 14))
    CLUS_2_VPU_TOP_AD_2 = bin_listTobin(CLUS_2_VPU_TOP_AD_2)
    CLUS_2_VPU_TOP_AD_2 = binTohex(CLUS_2_VPU_TOP_AD_2, 32)
    register_dict.append(CLUS_2_VPU_TOP_AD_2)
    # CLUS_2_VPU_TOP_SC_AD_3
    CLUS_2_VPU_TOP_SC_AD_3 = []
    CLUS_2_VPU_TOP_SC_AD_3.append(n_zeros_str(4))
    vpu_2_para_clus_2_scw_ad_3 = vpu_dict["vpu_2_para_clus_2_scw_ad_3"]
    CLUS_2_VPU_TOP_SC_AD_3.append(intToBin(vpu_2_para_clus_2_scw_ad_3, 14))
    vpu_2_para_clus_2_scr_ad_3 = vpu_dict["vpu_2_para_clus_2_scr_ad_3"]
    CLUS_2_VPU_TOP_SC_AD_3.append(intToBin(vpu_2_para_clus_2_scr_ad_3, 14))
    CLUS_2_VPU_TOP_SC_AD_3 = bin_listTobin(CLUS_2_VPU_TOP_SC_AD_3)
    CLUS_2_VPU_TOP_SC_AD_3 = binTohex(CLUS_2_VPU_TOP_SC_AD_3, 32)
    register_dict.append(CLUS_2_VPU_TOP_SC_AD_3)
    # CLUS_2_VPU_TOP_AD_3
    CLUS_2_VPU_TOP_AD_3 = []
    CLUS_2_VPU_TOP_AD_3.append(n_zeros_str(18))
    vpu_2_para_clus_2_ad_3 = vpu_dict["vpu_2_para_clus_2_ad_3"]
    CLUS_2_VPU_TOP_AD_3.append(intToBin(vpu_2_para_clus_2_ad_3, 14))
    CLUS_2_VPU_TOP_AD_3 = bin_listTobin(CLUS_2_VPU_TOP_AD_3)
    CLUS_2_VPU_TOP_AD_3 = binTohex(CLUS_2_VPU_TOP_AD_3, 32)
    register_dict.append(CLUS_2_VPU_TOP_AD_3)
    # CLUS_2_VPU_BLOCK_AD_JUMPER_0
    CLUS_2_VPU_BLOCK_AD_JUMPER_0 = []
    CLUS_2_VPU_BLOCK_AD_JUMPER_0.append(n_zeros_str(15))
    vpu_2_para_clus_2_block_ad_jump_0 = vpu_dict["vpu_2_para_clus_2_block_ad_jump_0"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_2_para_clus_2_block_ad_jump_0, 14))
    vpu_2_para_clus_2_block_ad_mode_enable = vpu_dict["vpu_2_para_clus_2_block_ad_mode_enable"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_2_para_clus_2_block_ad_mode_enable, 3))
    CLUS_2_VPU_BLOCK_AD_JUMPER_0 = bin_listTobin(CLUS_2_VPU_BLOCK_AD_JUMPER_0)
    CLUS_2_VPU_BLOCK_AD_JUMPER_0 = binTohex(CLUS_2_VPU_BLOCK_AD_JUMPER_0, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_AD_JUMPER_0)
    # CLUS_2_VPU_BLOCK_AD_JUMPER_1
    CLUS_2_VPU_BLOCK_AD_JUMPER_1 = []
    CLUS_2_VPU_BLOCK_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_2_para_clus_2_block_ad_jump_condit1 = vpu_dict["vpu_2_para_clus_2_block_ad_jump_condit1"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_ad_jump_condit1, 10))
    vpu_2_para_clus_2_block_ad_jump_condit0 = vpu_dict["vpu_2_para_clus_2_block_ad_jump_condit0"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_ad_jump_condit0, 10))
    CLUS_2_VPU_BLOCK_AD_JUMPER_1 = bin_listTobin(CLUS_2_VPU_BLOCK_AD_JUMPER_1)
    CLUS_2_VPU_BLOCK_AD_JUMPER_1 = binTohex(CLUS_2_VPU_BLOCK_AD_JUMPER_1, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_AD_JUMPER_1)
    # CLUS_2_VPU_BLOCK_AD_JUMPER_2
    CLUS_2_VPU_BLOCK_AD_JUMPER_2 = []
    CLUS_2_VPU_BLOCK_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_2_para_clus_2_block_ad_jump_2 = vpu_dict["vpu_2_para_clus_2_block_ad_jump_2"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_ad_jump_2, 14))
    vpu_2_para_clus_2_block_ad_jump_1 = vpu_dict["vpu_2_para_clus_2_block_ad_jump_1"]
    CLUS_2_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_ad_jump_1, 14))
    CLUS_2_VPU_BLOCK_AD_JUMPER_2 = bin_listTobin(CLUS_2_VPU_BLOCK_AD_JUMPER_2)
    CLUS_2_VPU_BLOCK_AD_JUMPER_2 = binTohex(CLUS_2_VPU_BLOCK_AD_JUMPER_2, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_AD_JUMPER_2)
    # CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0 = []
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_2_para_clus_2_block_scr_ad_jump_0 = vpu_dict["vpu_2_para_clus_2_block_scr_ad_jump_0"]
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0.append(intToBin(vpu_2_para_clus_2_block_scr_ad_jump_0, 14))
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0 = bin_listTobin(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0)
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0 = binTohex(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_0)
    # CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1 = []
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_2_para_clus_2_block_scr_ad_jump_condit1 = vpu_dict["vpu_2_para_clus_2_block_scr_ad_jump_condit1"]
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_scr_ad_jump_condit1, 10))
    vpu_2_para_clus_2_block_scr_ad_jump_condit0 = vpu_dict["vpu_2_para_clus_2_block_scr_ad_jump_condit0"]
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_scr_ad_jump_condit0, 10))
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1 = bin_listTobin(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1)
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1 = binTohex(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_1)
    # CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2 = []
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_2_para_clus_2_block_scr_ad_jump_2 = vpu_dict["vpu_2_para_clus_2_block_scr_ad_jump_2"]
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_scr_ad_jump_2, 14))
    vpu_2_para_clus_2_block_scr_ad_jump_1 = vpu_dict["vpu_2_para_clus_2_block_scr_ad_jump_1"]
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_scr_ad_jump_1, 14))
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2 = bin_listTobin(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2)
    CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2 = binTohex(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCR_AD_JUMPER_2)
    # CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0 = []
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_2_para_clus_2_block_scw_ad_jump_0 = vpu_dict["vpu_2_para_clus_2_block_scw_ad_jump_0"]
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0.append(intToBin(vpu_2_para_clus_2_block_scw_ad_jump_0, 14))
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0 = bin_listTobin(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0)
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0 = binTohex(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_0)
    # CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1 = []
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_2_para_clus_2_block_scw_ad_jump_condit1 = vpu_dict["vpu_2_para_clus_2_block_scw_ad_jump_condit1"]
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_scw_ad_jump_condit1, 10))
    vpu_2_para_clus_2_block_scw_ad_jump_condit0 = vpu_dict["vpu_2_para_clus_2_block_scw_ad_jump_condit0"]
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_2_para_clus_2_block_scw_ad_jump_condit0, 10))
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1 = bin_listTobin(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1)
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1 = binTohex(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_1)
    # CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2 = []
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_2_para_clus_2_block_scw_ad_jump_2 = vpu_dict["vpu_2_para_clus_2_block_scw_ad_jump_2"]
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_scw_ad_jump_2, 14))
    vpu_2_para_clus_2_block_scw_ad_jump_1 = vpu_dict["vpu_2_para_clus_2_block_scw_ad_jump_1"]
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_2_para_clus_2_block_scw_ad_jump_1, 14))
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2 = bin_listTobin(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2)
    CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2 = binTohex(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2, 32)
    register_dict.append(CLUS_2_VPU_BLOCK_SCW_AD_JUMPER_2)
    # CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX = []
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(n_zeros_str(13))
    vpu_2_para_clus_2_line_buffer_w_max = vpu_dict["vpu_2_para_clus_2_line_buffer_w_max"]
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_2_para_clus_2_line_buffer_w_max, 9))
    vpu_2_para_clus_2_line_buffer_h_max = vpu_dict["vpu_2_para_clus_2_line_buffer_h_max"]
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_2_para_clus_2_line_buffer_h_max, 10))
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX = bin_listTobin(CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX = binTohex(CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX, 32)
    register_dict.append(CLUS_2_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    # CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH = []
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH.append(n_zeros_str(26))
    vpu_2_para_clus_2_kernal_h = vpu_dict["vpu_2_para_clus_2_kernal_h"]
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_2_para_clus_2_kernal_h, 3))
    vpu_2_para_clus_2_kernal_h_stride = vpu_dict["vpu_2_para_clus_2_kernal_h_stride"]
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_2_para_clus_2_kernal_h_stride, 3))
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH = bin_listTobin(CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH)
    CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH = binTohex(CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH, 32)
    register_dict.append(CLUS_2_VPU_GLOBAL_LINE_BUFFER_KH)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1 = []
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_2_para_clus_2_output_l1_step = vpu_dict["vpu_2_para_clus_2_output_l1_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_output_l1_step, 9))
    vpu_2_para_clus_2_output_l1_condition = vpu_dict["vpu_2_para_clus_2_output_l1_condition"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_output_l1_condition, 9))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1 = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1 = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2 = []
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_2_para_clus_2_output_l2_step = vpu_dict["vpu_2_para_clus_2_output_l2_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_output_l2_step, 9))
    vpu_2_para_clus_2_output_l2_condition = vpu_dict["vpu_2_para_clus_2_output_l2_condition"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_output_l2_condition, 9))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2 = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2 = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3 = []
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_2_para_clus_2_output_l3_step = vpu_dict["vpu_2_para_clus_2_output_l3_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_output_l3_step, 9))
    vpu_2_para_clus_2_output_l3_condition = vpu_dict["vpu_2_para_clus_2_output_l3_condition"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_output_l3_condition, 9))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3 = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3 = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = []
    vpu_2_para_clus_2_output_l1_addr_step = vpu_dict["vpu_2_para_clus_2_output_l1_addr_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_2_para_clus_2_output_l1_addr_step, 32))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = []
    vpu_2_para_clus_2_output_l2_addr_step = vpu_dict["vpu_2_para_clus_2_output_l2_addr_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_2_para_clus_2_output_l2_addr_step, 32))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    # CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = []
    vpu_2_para_clus_2_output_l3_addr_step = vpu_dict["vpu_2_para_clus_2_output_l3_addr_step"]
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_2_para_clus_2_output_l3_addr_step, 32))
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = binTohex(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = []
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_2_para_clus_2_scw_l1_step = vpu_dict["vpu_2_para_clus_2_scw_l1_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_scw_l1_step, 9))
    vpu_2_para_clus_2_scw_l1_condition = vpu_dict["vpu_2_para_clus_2_scw_l1_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_scw_l1_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = []
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_2_para_clus_2_scw_l2_step = vpu_dict["vpu_2_para_clus_2_scw_l2_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_scw_l2_step, 9))
    vpu_2_para_clus_2_scw_l2_condition = vpu_dict["vpu_2_para_clus_2_scw_l2_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_scw_l2_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = []
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_2_para_clus_2_scw_l3_step = vpu_dict["vpu_2_para_clus_2_scw_l3_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_scw_l3_step, 9))
    vpu_2_para_clus_2_scw_l3_condition = vpu_dict["vpu_2_para_clus_2_scw_l3_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_scw_l3_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = []
    vpu_2_para_clus_2_scw_l1_addr_step = vpu_dict["vpu_2_para_clus_2_scw_l1_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_2_para_clus_2_scw_l1_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = []
    vpu_2_para_clus_2_scw_l2_addr_step = vpu_dict["vpu_2_para_clus_2_scw_l2_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_2_para_clus_2_scw_l2_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = []
    vpu_2_para_clus_2_scw_l3_addr_step = vpu_dict["vpu_2_para_clus_2_scw_l3_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_2_para_clus_2_scw_l3_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = []
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_2_para_clus_2_scr_l1_step = vpu_dict["vpu_2_para_clus_2_scr_l1_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_scr_l1_step, 9))
    vpu_2_para_clus_2_scr_l1_condition = vpu_dict["vpu_2_para_clus_2_scr_l1_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_2_para_clus_2_scr_l1_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = []
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_2_para_clus_2_scr_l2_step = vpu_dict["vpu_2_para_clus_2_scr_l2_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_scr_l2_step, 9))
    vpu_2_para_clus_2_scr_l2_condition = vpu_dict["vpu_2_para_clus_2_scr_l2_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_2_para_clus_2_scr_l2_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = []
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_2_para_clus_2_scr_l3_step = vpu_dict["vpu_2_para_clus_2_scr_l3_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_scr_l3_step, 9))
    vpu_2_para_clus_2_scr_l3_condition = vpu_dict["vpu_2_para_clus_2_scr_l3_condition"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_2_para_clus_2_scr_l3_condition, 9))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = []
    vpu_2_para_clus_2_scr_l1_addr_step = vpu_dict["vpu_2_para_clus_2_scr_l1_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_2_para_clus_2_scr_l1_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = []
    vpu_2_para_clus_2_scr_l2_addr_step = vpu_dict["vpu_2_para_clus_2_scr_l2_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_2_para_clus_2_scr_l2_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    # CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = []
    vpu_2_para_clus_2_scr_l3_addr_step = vpu_dict["vpu_2_para_clus_2_scr_l3_addr_step"]
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_2_para_clus_2_scr_l3_addr_step, 32))
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = binTohex(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_2_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    # CLUS_2_RS_WH_RATIO
    CLUS_2_RS_WH_RATIO = []
    vpu_2_para_i_clus_2_resize_param_width_ratio = vpu_dict["vpu_2_para_i_clus_2_resize_param_width_ratio"]
    CLUS_2_RS_WH_RATIO.append(intToBin(vpu_2_para_i_clus_2_resize_param_width_ratio, 16))
    vpu_2_para_i_clus_2_resize_param_height_ratio = vpu_dict["vpu_2_para_i_clus_2_resize_param_height_ratio"]
    CLUS_2_RS_WH_RATIO.append(intToBin(vpu_2_para_i_clus_2_resize_param_height_ratio, 16))
    CLUS_2_RS_WH_RATIO = bin_listTobin(CLUS_2_RS_WH_RATIO)
    CLUS_2_RS_WH_RATIO = binTohex(CLUS_2_RS_WH_RATIO, 32)
    register_dict.append(CLUS_2_RS_WH_RATIO)
    # CLUS_2_RS_INPUT_WH
    CLUS_2_RS_INPUT_WH = []
    CLUS_2_RS_INPUT_WH.append(n_zeros_str(12))
    vpu_2_para_i_clus_2_resize_param_input_width = vpu_dict["vpu_2_para_i_clus_2_resize_param_input_width"]
    CLUS_2_RS_INPUT_WH.append(intToBin(vpu_2_para_i_clus_2_resize_param_input_width, 9))
    vpu_2_para_i_clus_2_resize_param_input_height = vpu_dict["vpu_2_para_i_clus_2_resize_param_input_height"]
    CLUS_2_RS_INPUT_WH.append(intToBin(vpu_2_para_i_clus_2_resize_param_input_height, 11))
    CLUS_2_RS_INPUT_WH = bin_listTobin(CLUS_2_RS_INPUT_WH)
    CLUS_2_RS_INPUT_WH = binTohex(CLUS_2_RS_INPUT_WH, 32)
    register_dict.append(CLUS_2_RS_INPUT_WH)
    # CLUS_2_RS_OUTPUT_WH
    CLUS_2_RS_OUTPUT_WH = []
    CLUS_2_RS_OUTPUT_WH.append(n_zeros_str(10))
    vpu_2_para_i_clus_2_resize_param_output_width = vpu_dict["vpu_2_para_i_clus_2_resize_param_output_width"]
    CLUS_2_RS_OUTPUT_WH.append(intToBin(vpu_2_para_i_clus_2_resize_param_output_width, 11))
    vpu_2_para_i_clus_2_resize_param_output_height = vpu_dict["vpu_2_para_i_clus_2_resize_param_output_height"]
    CLUS_2_RS_OUTPUT_WH.append(intToBin(vpu_2_para_i_clus_2_resize_param_output_height, 11))
    CLUS_2_RS_OUTPUT_WH = bin_listTobin(CLUS_2_RS_OUTPUT_WH)
    CLUS_2_RS_OUTPUT_WH = binTohex(CLUS_2_RS_OUTPUT_WH, 32)
    register_dict.append(CLUS_2_RS_OUTPUT_WH)
    # CLUS_2_PL_INPUT_WH
    CLUS_2_PL_INPUT_WH = []
    CLUS_2_PL_INPUT_WH.append(n_zeros_str(12))
    vpu_2_para_i_clus_2_pooling_param_input_width = vpu_dict["vpu_2_para_i_clus_2_pooling_param_input_width"]
    CLUS_2_PL_INPUT_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_param_input_width, 9))
    vpu_2_para_i_clus_2_pooling_param_input_height = vpu_dict["vpu_2_para_i_clus_2_pooling_param_input_height"]
    CLUS_2_PL_INPUT_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_param_input_height, 11))
    CLUS_2_PL_INPUT_WH = bin_listTobin(CLUS_2_PL_INPUT_WH)
    CLUS_2_PL_INPUT_WH = binTohex(CLUS_2_PL_INPUT_WH, 32)
    register_dict.append(CLUS_2_PL_INPUT_WH)
    # CLUS_2_PL_OUTPUT_WH
    CLUS_2_PL_OUTPUT_WH = []
    CLUS_2_PL_OUTPUT_WH.append(n_zeros_str(10))
    vpu_2_para_i_clus_2_pooling_param_output_width = vpu_dict["vpu_2_para_i_clus_2_pooling_param_output_width"]
    CLUS_2_PL_OUTPUT_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_param_output_width, 11))
    vpu_2_para_i_clus_2_pooling_param_output_height = vpu_dict["vpu_2_para_i_clus_2_pooling_param_output_height"]
    CLUS_2_PL_OUTPUT_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_param_output_height, 11))
    CLUS_2_PL_OUTPUT_WH = bin_listTobin(CLUS_2_PL_OUTPUT_WH)
    CLUS_2_PL_OUTPUT_WH = binTohex(CLUS_2_PL_OUTPUT_WH, 32)
    register_dict.append(CLUS_2_PL_OUTPUT_WH)
    # CLUS_2_PL_PADDING_WH
    CLUS_2_PL_PADDING_WH = []
    CLUS_2_PL_PADDING_WH.append(n_zeros_str(6))
    vpu_2_para_i_clus_2_pooling_padding_mode = vpu_dict["vpu_2_para_i_clus_2_pooling_padding_mode"]
    CLUS_2_PL_PADDING_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_padding_mode, 4))
    vpu_2_para_i_clus_2_pooling_padding_width = vpu_dict["vpu_2_para_i_clus_2_pooling_padding_width"]
    CLUS_2_PL_PADDING_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_padding_width, 11))
    vpu_2_para_i_clus_2_pooling_padding_height = vpu_dict["vpu_2_para_i_clus_2_pooling_padding_height"]
    CLUS_2_PL_PADDING_WH.append(intToBin(vpu_2_para_i_clus_2_pooling_padding_height, 11))
    CLUS_2_PL_PADDING_WH = bin_listTobin(CLUS_2_PL_PADDING_WH)
    CLUS_2_PL_PADDING_WH = binTohex(CLUS_2_PL_PADDING_WH, 32)
    register_dict.append(CLUS_2_PL_PADDING_WH)
    # CLUS_3_VPU_TOP_SC_AD_0
    CLUS_3_VPU_TOP_SC_AD_0 = []
    CLUS_3_VPU_TOP_SC_AD_0.append(n_zeros_str(4))
    vpu_3_para_clus_3_scw_ad_0 = vpu_dict["vpu_3_para_clus_3_scw_ad_0"]
    CLUS_3_VPU_TOP_SC_AD_0.append(intToBin(vpu_3_para_clus_3_scw_ad_0, 14))
    vpu_3_para_clus_3_scr_ad_0 = vpu_dict["vpu_3_para_clus_3_scr_ad_0"]
    CLUS_3_VPU_TOP_SC_AD_0.append(intToBin(vpu_3_para_clus_3_scr_ad_0, 14))
    CLUS_3_VPU_TOP_SC_AD_0 = bin_listTobin(CLUS_3_VPU_TOP_SC_AD_0)
    CLUS_3_VPU_TOP_SC_AD_0 = binTohex(CLUS_3_VPU_TOP_SC_AD_0, 32)
    register_dict.append(CLUS_3_VPU_TOP_SC_AD_0)
    # CLUS_3_VPU_TOP_AD_0
    CLUS_3_VPU_TOP_AD_0 = []
    CLUS_3_VPU_TOP_AD_0.append(n_zeros_str(18))
    vpu_3_para_clus_3_ad_0 = vpu_dict["vpu_3_para_clus_3_ad_0"]
    CLUS_3_VPU_TOP_AD_0.append(intToBin(vpu_3_para_clus_3_ad_0, 14))
    CLUS_3_VPU_TOP_AD_0 = bin_listTobin(CLUS_3_VPU_TOP_AD_0)
    CLUS_3_VPU_TOP_AD_0 = binTohex(CLUS_3_VPU_TOP_AD_0, 32)
    register_dict.append(CLUS_3_VPU_TOP_AD_0)
    # CLUS_3_VPU_TOP_SC_AD_1
    CLUS_3_VPU_TOP_SC_AD_1 = []
    CLUS_3_VPU_TOP_SC_AD_1.append(n_zeros_str(4))
    vpu_3_para_clus_3_scw_ad_1 = vpu_dict["vpu_3_para_clus_3_scw_ad_1"]
    CLUS_3_VPU_TOP_SC_AD_1.append(intToBin(vpu_3_para_clus_3_scw_ad_1, 14))
    vpu_3_para_clus_3_scr_ad_1 = vpu_dict["vpu_3_para_clus_3_scr_ad_1"]
    CLUS_3_VPU_TOP_SC_AD_1.append(intToBin(vpu_3_para_clus_3_scr_ad_1, 14))
    CLUS_3_VPU_TOP_SC_AD_1 = bin_listTobin(CLUS_3_VPU_TOP_SC_AD_1)
    CLUS_3_VPU_TOP_SC_AD_1 = binTohex(CLUS_3_VPU_TOP_SC_AD_1, 32)
    register_dict.append(CLUS_3_VPU_TOP_SC_AD_1)
    # CLUS_3_VPU_TOP_AD_1
    CLUS_3_VPU_TOP_AD_1 = []
    CLUS_3_VPU_TOP_AD_1.append(n_zeros_str(18))
    vpu_3_para_clus_3_ad_1 = vpu_dict["vpu_3_para_clus_3_ad_1"]
    CLUS_3_VPU_TOP_AD_1.append(intToBin(vpu_3_para_clus_3_ad_1, 14))
    CLUS_3_VPU_TOP_AD_1 = bin_listTobin(CLUS_3_VPU_TOP_AD_1)
    CLUS_3_VPU_TOP_AD_1 = binTohex(CLUS_3_VPU_TOP_AD_1, 32)
    register_dict.append(CLUS_3_VPU_TOP_AD_1)
    # CLUS_3_VPU_TOP_SC_AD_2
    CLUS_3_VPU_TOP_SC_AD_2 = []
    CLUS_3_VPU_TOP_SC_AD_2.append(n_zeros_str(4))
    vpu_3_para_clus_3_scw_ad_2 = vpu_dict["vpu_3_para_clus_3_scw_ad_2"]
    CLUS_3_VPU_TOP_SC_AD_2.append(intToBin(vpu_3_para_clus_3_scw_ad_2, 14))
    vpu_3_para_clus_3_scr_ad_2 = vpu_dict["vpu_3_para_clus_3_scr_ad_2"]
    CLUS_3_VPU_TOP_SC_AD_2.append(intToBin(vpu_3_para_clus_3_scr_ad_2, 14))
    CLUS_3_VPU_TOP_SC_AD_2 = bin_listTobin(CLUS_3_VPU_TOP_SC_AD_2)
    CLUS_3_VPU_TOP_SC_AD_2 = binTohex(CLUS_3_VPU_TOP_SC_AD_2, 32)
    register_dict.append(CLUS_3_VPU_TOP_SC_AD_2)
    # CLUS_3_VPU_TOP_AD_2
    CLUS_3_VPU_TOP_AD_2 = []
    CLUS_3_VPU_TOP_AD_2.append(n_zeros_str(18))
    vpu_3_para_clus_3_ad_2 = vpu_dict["vpu_3_para_clus_3_ad_2"]
    CLUS_3_VPU_TOP_AD_2.append(intToBin(vpu_3_para_clus_3_ad_2, 14))
    CLUS_3_VPU_TOP_AD_2 = bin_listTobin(CLUS_3_VPU_TOP_AD_2)
    CLUS_3_VPU_TOP_AD_2 = binTohex(CLUS_3_VPU_TOP_AD_2, 32)
    register_dict.append(CLUS_3_VPU_TOP_AD_2)
    # CLUS_3_VPU_TOP_SC_AD_3
    CLUS_3_VPU_TOP_SC_AD_3 = []
    CLUS_3_VPU_TOP_SC_AD_3.append(n_zeros_str(4))
    vpu_3_para_clus_3_scw_ad_3 = vpu_dict["vpu_3_para_clus_3_scw_ad_3"]
    CLUS_3_VPU_TOP_SC_AD_3.append(intToBin(vpu_3_para_clus_3_scw_ad_3, 14))
    vpu_3_para_clus_3_scr_ad_3 = vpu_dict["vpu_3_para_clus_3_scr_ad_3"]
    CLUS_3_VPU_TOP_SC_AD_3.append(intToBin(vpu_3_para_clus_3_scr_ad_3, 14))
    CLUS_3_VPU_TOP_SC_AD_3 = bin_listTobin(CLUS_3_VPU_TOP_SC_AD_3)
    CLUS_3_VPU_TOP_SC_AD_3 = binTohex(CLUS_3_VPU_TOP_SC_AD_3, 32)
    register_dict.append(CLUS_3_VPU_TOP_SC_AD_3)
    # CLUS_3_VPU_TOP_AD_3
    CLUS_3_VPU_TOP_AD_3 = []
    CLUS_3_VPU_TOP_AD_3.append(n_zeros_str(18))
    vpu_3_para_clus_3_ad_3 = vpu_dict["vpu_3_para_clus_3_ad_3"]
    CLUS_3_VPU_TOP_AD_3.append(intToBin(vpu_3_para_clus_3_ad_3, 14))
    CLUS_3_VPU_TOP_AD_3 = bin_listTobin(CLUS_3_VPU_TOP_AD_3)
    CLUS_3_VPU_TOP_AD_3 = binTohex(CLUS_3_VPU_TOP_AD_3, 32)
    register_dict.append(CLUS_3_VPU_TOP_AD_3)
    # CLUS_3_VPU_BLOCK_AD_JUMPER_0
    CLUS_3_VPU_BLOCK_AD_JUMPER_0 = []
    CLUS_3_VPU_BLOCK_AD_JUMPER_0.append(n_zeros_str(15))
    vpu_3_para_clus_3_block_ad_jump_0 = vpu_dict["vpu_3_para_clus_3_block_ad_jump_0"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_3_para_clus_3_block_ad_jump_0, 14))
    vpu_3_para_clus_3_block_ad_mode_enable = vpu_dict["vpu_3_para_clus_3_block_ad_mode_enable"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_0.append(intToBin(vpu_3_para_clus_3_block_ad_mode_enable, 3))
    CLUS_3_VPU_BLOCK_AD_JUMPER_0 = bin_listTobin(CLUS_3_VPU_BLOCK_AD_JUMPER_0)
    CLUS_3_VPU_BLOCK_AD_JUMPER_0 = binTohex(CLUS_3_VPU_BLOCK_AD_JUMPER_0, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_AD_JUMPER_0)
    # CLUS_3_VPU_BLOCK_AD_JUMPER_1
    CLUS_3_VPU_BLOCK_AD_JUMPER_1 = []
    CLUS_3_VPU_BLOCK_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_3_para_clus_3_block_ad_jump_condit1 = vpu_dict["vpu_3_para_clus_3_block_ad_jump_condit1"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_ad_jump_condit1, 10))
    vpu_3_para_clus_3_block_ad_jump_condit0 = vpu_dict["vpu_3_para_clus_3_block_ad_jump_condit0"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_ad_jump_condit0, 10))
    CLUS_3_VPU_BLOCK_AD_JUMPER_1 = bin_listTobin(CLUS_3_VPU_BLOCK_AD_JUMPER_1)
    CLUS_3_VPU_BLOCK_AD_JUMPER_1 = binTohex(CLUS_3_VPU_BLOCK_AD_JUMPER_1, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_AD_JUMPER_1)
    # CLUS_3_VPU_BLOCK_AD_JUMPER_2
    CLUS_3_VPU_BLOCK_AD_JUMPER_2 = []
    CLUS_3_VPU_BLOCK_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_3_para_clus_3_block_ad_jump_2 = vpu_dict["vpu_3_para_clus_3_block_ad_jump_2"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_ad_jump_2, 14))
    vpu_3_para_clus_3_block_ad_jump_1 = vpu_dict["vpu_3_para_clus_3_block_ad_jump_1"]
    CLUS_3_VPU_BLOCK_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_ad_jump_1, 14))
    CLUS_3_VPU_BLOCK_AD_JUMPER_2 = bin_listTobin(CLUS_3_VPU_BLOCK_AD_JUMPER_2)
    CLUS_3_VPU_BLOCK_AD_JUMPER_2 = binTohex(CLUS_3_VPU_BLOCK_AD_JUMPER_2, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_AD_JUMPER_2)
    # CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0 = []
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_3_para_clus_3_block_scr_ad_jump_0 = vpu_dict["vpu_3_para_clus_3_block_scr_ad_jump_0"]
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0.append(intToBin(vpu_3_para_clus_3_block_scr_ad_jump_0, 14))
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0 = bin_listTobin(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0)
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0 = binTohex(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_0)
    # CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1 = []
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_3_para_clus_3_block_scr_ad_jump_condit1 = vpu_dict["vpu_3_para_clus_3_block_scr_ad_jump_condit1"]
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_scr_ad_jump_condit1, 10))
    vpu_3_para_clus_3_block_scr_ad_jump_condit0 = vpu_dict["vpu_3_para_clus_3_block_scr_ad_jump_condit0"]
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_scr_ad_jump_condit0, 10))
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1 = bin_listTobin(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1)
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1 = binTohex(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_1)
    # CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2 = []
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_3_para_clus_3_block_scr_ad_jump_2 = vpu_dict["vpu_3_para_clus_3_block_scr_ad_jump_2"]
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_scr_ad_jump_2, 14))
    vpu_3_para_clus_3_block_scr_ad_jump_1 = vpu_dict["vpu_3_para_clus_3_block_scr_ad_jump_1"]
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_scr_ad_jump_1, 14))
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2 = bin_listTobin(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2)
    CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2 = binTohex(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCR_AD_JUMPER_2)
    # CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0 = []
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0.append(n_zeros_str(18))
    vpu_3_para_clus_3_block_scw_ad_jump_0 = vpu_dict["vpu_3_para_clus_3_block_scw_ad_jump_0"]
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0.append(intToBin(vpu_3_para_clus_3_block_scw_ad_jump_0, 14))
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0 = bin_listTobin(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0)
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0 = binTohex(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_0)
    # CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1 = []
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1.append(n_zeros_str(12))
    vpu_3_para_clus_3_block_scw_ad_jump_condit1 = vpu_dict["vpu_3_para_clus_3_block_scw_ad_jump_condit1"]
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_scw_ad_jump_condit1, 10))
    vpu_3_para_clus_3_block_scw_ad_jump_condit0 = vpu_dict["vpu_3_para_clus_3_block_scw_ad_jump_condit0"]
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1.append(intToBin(vpu_3_para_clus_3_block_scw_ad_jump_condit0, 10))
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1 = bin_listTobin(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1)
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1 = binTohex(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_1)
    # CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2 = []
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2.append(n_zeros_str(4))
    vpu_3_para_clus_3_block_scw_ad_jump_2 = vpu_dict["vpu_3_para_clus_3_block_scw_ad_jump_2"]
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_scw_ad_jump_2, 14))
    vpu_3_para_clus_3_block_scw_ad_jump_1 = vpu_dict["vpu_3_para_clus_3_block_scw_ad_jump_1"]
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2.append(intToBin(vpu_3_para_clus_3_block_scw_ad_jump_1, 14))
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2 = bin_listTobin(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2)
    CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2 = binTohex(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2, 32)
    register_dict.append(CLUS_3_VPU_BLOCK_SCW_AD_JUMPER_2)
    # CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX = []
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(n_zeros_str(13))
    vpu_3_para_clus_3_line_buffer_w_max = vpu_dict["vpu_3_para_clus_3_line_buffer_w_max"]
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_3_para_clus_3_line_buffer_w_max, 9))
    vpu_3_para_clus_3_line_buffer_h_max = vpu_dict["vpu_3_para_clus_3_line_buffer_h_max"]
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX.append(intToBin(vpu_3_para_clus_3_line_buffer_h_max, 10))
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX = bin_listTobin(CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX = binTohex(CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX, 32)
    register_dict.append(CLUS_3_VPU_GLOBAL_LINE_BUFFER_WH_MAX)
    # CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH = []
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH.append(n_zeros_str(26))
    vpu_3_para_clus_3_kernal_h = vpu_dict["vpu_3_para_clus_3_kernal_h"]
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_3_para_clus_3_kernal_h, 3))
    vpu_3_para_clus_3_kernal_h_stride = vpu_dict["vpu_3_para_clus_3_kernal_h_stride"]
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH.append(intToBin(vpu_3_para_clus_3_kernal_h_stride, 3))
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH = bin_listTobin(CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH)
    CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH = binTohex(CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH, 32)
    register_dict.append(CLUS_3_VPU_GLOBAL_LINE_BUFFER_KH)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1 = []
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_3_para_clus_3_output_l1_step = vpu_dict["vpu_3_para_clus_3_output_l1_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_output_l1_step, 9))
    vpu_3_para_clus_3_output_l1_condition = vpu_dict["vpu_3_para_clus_3_output_l1_condition"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_output_l1_condition, 9))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1 = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1 = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2 = []
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_3_para_clus_3_output_l2_step = vpu_dict["vpu_3_para_clus_3_output_l2_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_output_l2_step, 9))
    vpu_3_para_clus_3_output_l2_condition = vpu_dict["vpu_3_para_clus_3_output_l2_condition"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_output_l2_condition, 9))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2 = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2 = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3 = []
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_3_para_clus_3_output_l3_step = vpu_dict["vpu_3_para_clus_3_output_l3_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_output_l3_step, 9))
    vpu_3_para_clus_3_output_l3_condition = vpu_dict["vpu_3_para_clus_3_output_l3_condition"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_output_l3_condition, 9))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3 = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3 = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = []
    vpu_3_para_clus_3_output_l1_addr_step = vpu_dict["vpu_3_para_clus_3_output_l1_addr_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_3_para_clus_3_output_l1_addr_step, 32))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_1_AD_STEP)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = []
    vpu_3_para_clus_3_output_l2_addr_step = vpu_dict["vpu_3_para_clus_3_output_l2_addr_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_3_para_clus_3_output_l2_addr_step, 32))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_2_AD_STEP)
    # CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = []
    vpu_3_para_clus_3_output_l3_addr_step = vpu_dict["vpu_3_para_clus_3_output_l3_addr_step"]
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_3_para_clus_3_output_l3_addr_step, 32))
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP = binTohex(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_OUTPUT_FOR_LOOP_3_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = []
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_3_para_clus_3_scw_l1_step = vpu_dict["vpu_3_para_clus_3_scw_l1_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_scw_l1_step, 9))
    vpu_3_para_clus_3_scw_l1_condition = vpu_dict["vpu_3_para_clus_3_scw_l1_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_scw_l1_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = []
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_3_para_clus_3_scw_l2_step = vpu_dict["vpu_3_para_clus_3_scw_l2_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_scw_l2_step, 9))
    vpu_3_para_clus_3_scw_l2_condition = vpu_dict["vpu_3_para_clus_3_scw_l2_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_scw_l2_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = []
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_3_para_clus_3_scw_l3_step = vpu_dict["vpu_3_para_clus_3_scw_l3_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_scw_l3_step, 9))
    vpu_3_para_clus_3_scw_l3_condition = vpu_dict["vpu_3_para_clus_3_scw_l3_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_scw_l3_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = []
    vpu_3_para_clus_3_scw_l1_addr_step = vpu_dict["vpu_3_para_clus_3_scw_l1_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_3_para_clus_3_scw_l1_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_1_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = []
    vpu_3_para_clus_3_scw_l2_addr_step = vpu_dict["vpu_3_para_clus_3_scw_l2_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_3_para_clus_3_scw_l2_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_2_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = []
    vpu_3_para_clus_3_scw_l3_addr_step = vpu_dict["vpu_3_para_clus_3_scw_l3_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_3_para_clus_3_scw_l3_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_W_FOR_LOOP_3_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = []
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(n_zeros_str(14))
    vpu_3_para_clus_3_scr_l1_step = vpu_dict["vpu_3_para_clus_3_scr_l1_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_scr_l1_step, 9))
    vpu_3_para_clus_3_scr_l1_condition = vpu_dict["vpu_3_para_clus_3_scr_l1_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1.append(intToBin(vpu_3_para_clus_3_scr_l1_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = []
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(n_zeros_str(14))
    vpu_3_para_clus_3_scr_l2_step = vpu_dict["vpu_3_para_clus_3_scr_l2_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_scr_l2_step, 9))
    vpu_3_para_clus_3_scr_l2_condition = vpu_dict["vpu_3_para_clus_3_scr_l2_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2.append(intToBin(vpu_3_para_clus_3_scr_l2_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = []
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(n_zeros_str(14))
    vpu_3_para_clus_3_scr_l3_step = vpu_dict["vpu_3_para_clus_3_scr_l3_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_scr_l3_step, 9))
    vpu_3_para_clus_3_scr_l3_condition = vpu_dict["vpu_3_para_clus_3_scr_l3_condition"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3.append(intToBin(vpu_3_para_clus_3_scr_l3_condition, 9))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3 = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = []
    vpu_3_para_clus_3_scr_l1_addr_step = vpu_dict["vpu_3_para_clus_3_scr_l1_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP.append(intToBin(vpu_3_para_clus_3_scr_l1_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_1_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = []
    vpu_3_para_clus_3_scr_l2_addr_step = vpu_dict["vpu_3_para_clus_3_scr_l2_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP.append(intToBin(vpu_3_para_clus_3_scr_l2_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_2_AD_STEP)
    # CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = []
    vpu_3_para_clus_3_scr_l3_addr_step = vpu_dict["vpu_3_para_clus_3_scr_l3_addr_step"]
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP.append(intToBin(vpu_3_para_clus_3_scr_l3_addr_step, 32))
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = bin_listTobin(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP = binTohex(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP, 32)
    register_dict.append(CLUS_3_VPU_TOP_SHORT_CUT_R_FOR_LOOP_3_AD_STEP)
    # CLUS_3_RS_WH_RATIO
    CLUS_3_RS_WH_RATIO = []
    vpu_3_para_i_clus_3_resize_param_width_ratio = vpu_dict["vpu_3_para_i_clus_3_resize_param_width_ratio"]
    CLUS_3_RS_WH_RATIO.append(intToBin(vpu_3_para_i_clus_3_resize_param_width_ratio, 16))
    vpu_3_para_i_clus_3_resize_param_height_ratio = vpu_dict["vpu_3_para_i_clus_3_resize_param_height_ratio"]
    CLUS_3_RS_WH_RATIO.append(intToBin(vpu_3_para_i_clus_3_resize_param_height_ratio, 16))
    CLUS_3_RS_WH_RATIO = bin_listTobin(CLUS_3_RS_WH_RATIO)
    CLUS_3_RS_WH_RATIO = binTohex(CLUS_3_RS_WH_RATIO, 32)
    register_dict.append(CLUS_3_RS_WH_RATIO)
    # CLUS_3_RS_INPUT_WH
    CLUS_3_RS_INPUT_WH = []
    CLUS_3_RS_INPUT_WH.append(n_zeros_str(12))
    vpu_3_para_i_clus_3_resize_param_input_width = vpu_dict["vpu_3_para_i_clus_3_resize_param_input_width"]
    CLUS_3_RS_INPUT_WH.append(intToBin(vpu_3_para_i_clus_3_resize_param_input_width, 9))
    vpu_3_para_i_clus_3_resize_param_input_height = vpu_dict["vpu_3_para_i_clus_3_resize_param_input_height"]
    CLUS_3_RS_INPUT_WH.append(intToBin(vpu_3_para_i_clus_3_resize_param_input_height, 11))
    CLUS_3_RS_INPUT_WH = bin_listTobin(CLUS_3_RS_INPUT_WH)
    CLUS_3_RS_INPUT_WH = binTohex(CLUS_3_RS_INPUT_WH, 32)
    register_dict.append(CLUS_3_RS_INPUT_WH)
    # CLUS_3_RS_OUTPUT_WH
    CLUS_3_RS_OUTPUT_WH = []
    CLUS_3_RS_OUTPUT_WH.append(n_zeros_str(10))
    vpu_3_para_i_clus_3_resize_param_output_width = vpu_dict["vpu_3_para_i_clus_3_resize_param_output_width"]
    CLUS_3_RS_OUTPUT_WH.append(intToBin(vpu_3_para_i_clus_3_resize_param_output_width, 11))
    vpu_3_para_i_clus_3_resize_param_output_height = vpu_dict["vpu_3_para_i_clus_3_resize_param_output_height"]
    CLUS_3_RS_OUTPUT_WH.append(intToBin(vpu_3_para_i_clus_3_resize_param_output_height, 11))
    CLUS_3_RS_OUTPUT_WH = bin_listTobin(CLUS_3_RS_OUTPUT_WH)
    CLUS_3_RS_OUTPUT_WH = binTohex(CLUS_3_RS_OUTPUT_WH, 32)
    register_dict.append(CLUS_3_RS_OUTPUT_WH)
    # CLUS_3_PL_INPUT_WH
    CLUS_3_PL_INPUT_WH = []
    CLUS_3_PL_INPUT_WH.append(n_zeros_str(12))
    vpu_3_para_i_clus_3_pooling_param_input_width = vpu_dict["vpu_3_para_i_clus_3_pooling_param_input_width"]
    CLUS_3_PL_INPUT_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_param_input_width, 9))
    vpu_3_para_i_clus_3_pooling_param_input_height = vpu_dict["vpu_3_para_i_clus_3_pooling_param_input_height"]
    CLUS_3_PL_INPUT_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_param_input_height, 11))
    CLUS_3_PL_INPUT_WH = bin_listTobin(CLUS_3_PL_INPUT_WH)
    CLUS_3_PL_INPUT_WH = binTohex(CLUS_3_PL_INPUT_WH, 32)
    register_dict.append(CLUS_3_PL_INPUT_WH)
    # CLUS_3_PL_OUTPUT_WH
    CLUS_3_PL_OUTPUT_WH = []
    CLUS_3_PL_OUTPUT_WH.append(n_zeros_str(10))
    vpu_3_para_i_clus_3_pooling_param_output_width = vpu_dict["vpu_3_para_i_clus_3_pooling_param_output_width"]
    CLUS_3_PL_OUTPUT_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_param_output_width, 11))
    vpu_3_para_i_clus_3_pooling_param_output_height = vpu_dict["vpu_3_para_i_clus_3_pooling_param_output_height"]
    CLUS_3_PL_OUTPUT_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_param_output_height, 11))
    CLUS_3_PL_OUTPUT_WH = bin_listTobin(CLUS_3_PL_OUTPUT_WH)
    CLUS_3_PL_OUTPUT_WH = binTohex(CLUS_3_PL_OUTPUT_WH, 32)
    register_dict.append(CLUS_3_PL_OUTPUT_WH)
    # CLUS_3_PL_PADDING_WH
    CLUS_3_PL_PADDING_WH = []
    CLUS_3_PL_PADDING_WH.append(n_zeros_str(6))
    vpu_3_para_i_clus_3_pooling_padding_mode = vpu_dict["vpu_3_para_i_clus_3_pooling_padding_mode"]
    CLUS_3_PL_PADDING_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_padding_mode, 4))
    vpu_3_para_i_clus_3_pooling_padding_width = vpu_dict["vpu_3_para_i_clus_3_pooling_padding_width"]
    CLUS_3_PL_PADDING_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_padding_width, 11))
    vpu_3_para_i_clus_3_pooling_padding_height = vpu_dict["vpu_3_para_i_clus_3_pooling_padding_height"]
    CLUS_3_PL_PADDING_WH.append(intToBin(vpu_3_para_i_clus_3_pooling_padding_height, 11))
    CLUS_3_PL_PADDING_WH = bin_listTobin(CLUS_3_PL_PADDING_WH)
    CLUS_3_PL_PADDING_WH = binTohex(CLUS_3_PL_PADDING_WH, 32)
    register_dict.append(CLUS_3_PL_PADDING_WH)
    return register_dict


if __name__ == "__main__":
    ADDR = ["32'h00009000", "32'h00009004", "32'h00009008", "32'h0000900C", "32'h00009010", "32'h00009014",
            "32'h00009018", "32'h0000901C", "32'h00009020", "32'h00009024", "32'h00009028", "32'h0000902C",
            "32'h00009030", "32'h00009034", "32'h00009038", "32'h0000903C", "32'h00009040", "32'h00009044",
            "32'h00009048", "32'h0000904C", "32'h00009050", "32'h00009054", "32'h00009058", "32'h0000905C",
            "32'h00009060", "32'h00009064", "32'h00009068", "32'h0000906C", "32'h00009070", "32'h00009074",
            "32'h00009078", "32'h0000907C", "32'h00009080", "32'h00009084", "32'h00009088", "32'h0000908C",
            "32'h00009090", "32'h00009094", "32'h00009098", "32'h0000909C", "32'h000090A0", "32'h000090A4",
            "32'h000090A8", "32'h000090AC", "32'h000090B0", "32'h000090B8", "32'h000090B4", "32'h000090BC",
            "32'h000090C0", "32'h000090C4", "32'h000090C8", "32'h000090CC", "32'h000090D0", "32'h000090D4",
            "32'h000090D8", "32'h000090DC", "32'h000090E0", "32'h000090E4", "32'h000090E8", "32'h000090EC",
            "32'h000090F0", "32'h000090F4", "32'h000090F8", "32'h000090FC", "32'h00009150", "32'h00009154",
            "32'h00009158", "32'h0000915C", "32'h00009160", "32'h00009164", "32'h00009168", "32'h0000916C",
            "32'h00009170", "32'h00009174", "32'h00009178", "32'h0000917C", "32'h00009180", "32'h00009184",
            "32'h00009188", "32'h0000918C", "32'h00009190", "32'h00009194", "32'h00009198", "32'h0000919C",
            "32'h000091A0", "32'h000091A4", "32'h000091A8", "32'h000091AC", "32'h000091B0", "32'h000091B4",
            "32'h000091B8", "32'h000091BC", "32'h000091C0", "32'h000091C4", "32'h000091C8", "32'h000091CC",
            "32'h000091D0", "32'h000091D4", "32'h000091D8", "32'h000091DC", "32'h000091E0", "32'h000091E4",
            "32'h000091E8", "32'h000091EC", "32'h000091F0", "32'h000091F4", "32'h000091F8", "32'h00009250",
            "32'h00009254", "32'h00009258", "32'h0000925C", "32'h00009260", "32'h00009264", "32'h00009268",
            "32'h0000926C", "32'h00009270", "32'h00009274", "32'h00009278", "32'h0000927C", "32'h00009280",
            "32'h00009284", "32'h00009288", "32'h0000928C", "32'h00009290", "32'h00009294", "32'h00009298",
            "32'h0000929C", "32'h000092A0", "32'h000092A4", "32'h000092A8", "32'h000092AC", "32'h000092B0",
            "32'h000092B4", "32'h000092B8", "32'h000092BC", "32'h000092C0", "32'h000092C4", "32'h000092C8",
            "32'h000092CC", "32'h000092D0", "32'h000092D4", "32'h000092D8", "32'h000092DC", "32'h000092E0",
            "32'h000092E4", "32'h000092E8", "32'h000092EC", "32'h000092F0", "32'h000092F4", "32'h000092F8",
            "32'h00009450", "32'h00009454", "32'h00009458", "32'h0000945C", "32'h00009460", "32'h00009464",
            "32'h00009468", "32'h0000946C", "32'h00009470", "32'h00009474", "32'h00009478", "32'h0000947C",
            "32'h00009480", "32'h00009484", "32'h00009488", "32'h0000948C", "32'h00009490", "32'h00009494",
            "32'h00009498", "32'h0000949C", "32'h000094a0", "32'h000094a4", "32'h000094a8", "32'h000094aC",
            "32'h000094b0", "32'h000094b4", "32'h000094b8", "32'h000094bC", "32'h000094c0", "32'h000094c4",
            "32'h000094c8", "32'h000094cC", "32'h000094d0", "32'h000094d4", "32'h000094d8", "32'h000094dC",
            "32'h000094e0", "32'h000094e4", "32'h000094e8", "32'h000094eC", "32'h000094f0", "32'h000094f4",
            "32'h000094f8", "32'h00009850", "32'h00009854", "32'h00009858", "32'h0000985C", "32'h00009860",
            "32'h00009864", "32'h00009868", "32'h0000986C", "32'h00009870", "32'h00009874", "32'h00009878",
            "32'h0000987C", "32'h00009880", "32'h00009884", "32'h00009888", "32'h0000988C", "32'h00009890",
            "32'h00009894", "32'h00009898", "32'h0000989C", "32'h000098a0", "32'h000098a4", "32'h000098a8",
            "32'h000098aC", "32'h000098b0", "32'h000098b4", "32'h000098b8", "32'h000098bC", "32'h000098c0",
            "32'h000098c4", "32'h000098c8", "32'h000098cC", "32'h000098d0", "32'h000098d4", "32'h000098d8",
            "32'h000098dC", "32'h000098e0", "32'h000098e4", "32'h000098e8", "32'h000098eC", "32'h000098f0",
            "32'h000098f4", "32'h000098f8"]
    para_buf = get_list(nums=432, val=0)
    para_buf[VpuRegister.vpu_SF_para_vpu_fifo_group_sf_rst] = 0
    para_buf[VpuRegister.vpu_SF_para_global_line_buffer_sf_rst] = 0
    para_buf[VpuRegister.vpu_SF_para_sc_buffer_sf_rst] = 0
    para_buf[VpuRegister.vpu_SF_para_vpu_unit_sf_rst] = 0
    para_buf[VpuRegister.vpu_SF_para_interface_sf_rst] = 0
    para_buf[VpuRegister.vpu_SF_para_top_ctrl_sf_rst] = 0
    para_buf[VpuRegister.vpu_TEST_para_done_len] = 0
    para_buf[VpuRegister.vpu_TEST_para_bp_mode] = 0
    para_buf[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_sel] = 0
    para_buf[VpuRegister.vpu_TEST_para_vpu_unit_test_mode_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_odd_output_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_bp_mode_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_short_st_arb_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_psum_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_short_cut_buffer] = 0
    para_buf[VpuRegister.vpu_SF_para_line_controller_3] = 0
    para_buf[VpuRegister.vpu_SF_para_line_controller_2] = 0
    para_buf[VpuRegister.vpu_SF_para_line_controller_1] = 0
    para_buf[VpuRegister.vpu_SF_para_line_controller_0] = 0
    para_buf[VpuRegister.vpu_SF_para_fifo2line_buffer_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_global_line_buffer_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_input_fifo_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_vpu_sc_mode] = 0
    para_buf[VpuRegister.vpu_SF_para_vpu_sc_enable] = 0
    para_buf[VpuRegister.vpu_SF_para_vpu_enable] = 0
    para_buf[VpuRegister.vpu_TOP_para_cim_weights_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_cluster_weights_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_interface_write_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_sc_width] = 0
    para_buf[VpuRegister.vpu_TOP_para_switch_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_sc_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_line_buffer_mode] = 0
    para_buf[VpuRegister.vpu_TOP_para_fmt_channel_type] = 0
    para_buf[VpuRegister.vpu_TOP_para_read_line_buffer_mode] = 0
    para_buf[VpuRegister.vpu_WISE_para_quantize_min] = 0
    para_buf[VpuRegister.vpu_WISE_para_quantize_max] = 0
    para_buf[VpuRegister.vpu_WISE_para_mode] = 0
    para_buf[VpuRegister.vpu_WISE_para_quantize_mul] = 0
    para_buf[VpuRegister.vpu_WISE_para_quantize_shf] = 0
    para_buf[VpuRegister.vpu_WISE_para_quantize_off] = 0
    para_buf[VpuRegister.vpu_WISE_para_element_wise_dequantize_0_sclale_o] = 0
    para_buf[VpuRegister.vpu_WISE_para_element_wise_dequantize_0_shifter_o] = 0
    para_buf[VpuRegister.vpu_WISE_para_dequantize_0_off] = 0
    para_buf[VpuRegister.vpu_WISE_para_element_wise_dequantize_1_sclale_o] = 0
    para_buf[VpuRegister.vpu_WISE_para_element_wise_dequantize_1_shifter_o] = 0
    para_buf[VpuRegister.vpu_WISE_para_dequantize_1_off] = 0
    para_buf[VpuRegister.vpu_WISE_para_div_fix_param] = 0
    para_buf[VpuRegister.vpu_WISE_para_div_shifter] = 0
    para_buf[VpuRegister.vpu_BASIC_para_i_resize_param_half_pixal_flag] = 0
    para_buf[VpuRegister.vpu_BASIC_para_i_resize_param_bil_nn_sel_flag] = 0
    para_buf[VpuRegister.vpu_BASIC_para_pl_func_mode] = 0
    para_buf[VpuRegister.vpu_BASIC_para_pl_factor] = 0
    para_buf[VpuRegister.vpu_FILTER_para_i_pooling_filter_width] = 0
    para_buf[VpuRegister.vpu_FILTER_para_i_pooling_filter_height] = 0
    para_buf[VpuRegister.vpu_STRIDE_para_i_pooling_stride_width] = 0
    para_buf[VpuRegister.vpu_STRIDE_para_i_pooling_stride_height] = 0
    para_buf[VpuRegister.vpu_BASIC_para_i_fmt_width] = 0
    para_buf[VpuRegister.vpu_BASIC_para_i_pl_rs_sel] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b0_ad] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b1_ad] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b2_ad] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b3_ad] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b4_wt_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b5_wt_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b6_wt_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b7_wt_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b4_rd_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b5_rd_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b6_rd_addr] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_b7_rd_addr] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_stov] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_ema] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_emaw] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_emas] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_ret1n] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawl] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_rawlm] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_wabl] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_sc_buffer_wablm] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_stov] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_ema] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_emaw] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_emas] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_ret1n] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_rawl] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_rawlm] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_wabl] = 0
    para_buf[VpuRegister.vpu_GLOBAL_para_global_buffer_wablm] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_stov] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_ema] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_emaw] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_emas] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_ret1n] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_rawl] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_rawlm] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_wabl] = 0
    para_buf[VpuRegister.vpu_LINE_para_line_buffer_wablm] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_stov] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_ema] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_emaw] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_emas] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_ret1n] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_rawl] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_rawlm] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_wabl] = 0
    para_buf[VpuRegister.vpu_LUT_para_lut_wablm] = 0
    para_buf[VpuRegister.vpu_UNIT_para_i_vpu_unit_seed] = 0
    para_buf[VpuRegister.vpu_INTERFACE_para_i_vpu_interface_seed] = 0
    para_buf[VpuRegister.vpu_IN_para_i_vpu_in_fifo_group_seed] = 0
    para_buf[VpuRegister.vpu_DATA_para_i_target_data_amount] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_ans_data_sc] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_ans_data_wo_q] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_ans_done_flag] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_ans_col_done] = 0
    para_buf[VpuRegister.vpu_BIST_para_all_ans_data_out_psum] = 0
    para_buf[VpuRegister.vpu_BIST_para_all_ans_data_out_0] = 0
    para_buf[VpuRegister.vpu_BIST_para_all_ans_data_out_1] = 0
    para_buf[VpuRegister.vpu_BIST_para_all_ans_data_out_2] = 0
    para_buf[VpuRegister.vpu_BIST_para_all_ans_data_out_3] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b0_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b0_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b0_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b1_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b1_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b1_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b2_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b2_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b2_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b3_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b3_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b3_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b4_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b4_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b4_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b5_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b5_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b5_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b6_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b6_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b6_ans_addr] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b7_ans_data] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b7_ans_ctrl] = 0
    para_buf[VpuRegister.vpu_BIST_para_o_b7_ans_addr] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_ad_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_ad_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_ad_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_ad_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_ad_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_ad_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_ad_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_ad_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_ad_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_ad_3] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_ad_3] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_ad_3] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_mode_enable] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scr_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_block_scw_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_line_buffer_w_max] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_line_buffer_h_max] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_kernal_h] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_kernal_h_stride] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l1_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l1_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l2_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l2_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l3_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l3_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_output_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l1_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l1_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l2_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l2_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l3_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l3_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scw_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l1_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l1_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l2_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l2_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l3_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l3_condition] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_clus_0_scr_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_width_ratio] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_height_ratio] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_width] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_input_height] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_output_width] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_resize_param_output_height] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_width] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_param_input_height] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_width] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_param_output_height] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_mode] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_width] = 0
    para_buf[VpuRegister.vpu_0_para_i_clus_0_pooling_padding_height] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_ad_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_ad_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_ad_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_ad_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_ad_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_ad_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_ad_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_ad_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_ad_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_ad_3] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_ad_3] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_ad_3] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_mode_enable] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scr_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_block_scw_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_line_buffer_w_max] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_line_buffer_h_max] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_kernal_h] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_kernal_h_stride] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l1_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l1_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l2_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l2_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l3_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l3_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_output_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l1_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l1_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l2_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l2_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l3_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l3_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scw_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l1_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l1_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l2_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l2_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l3_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l3_condition] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_clus_1_scr_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_width_ratio] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_height_ratio] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_width] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_input_height] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_output_width] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_resize_param_output_height] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_width] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_param_input_height] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_width] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_param_output_height] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_mode] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_width] = 0
    para_buf[VpuRegister.vpu_1_para_i_clus_1_pooling_padding_height] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_ad_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_ad_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_ad_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_ad_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_ad_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_ad_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_ad_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_ad_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_ad_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_ad_3] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_ad_3] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_ad_3] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_mode_enable] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scr_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_block_scw_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_line_buffer_w_max] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_line_buffer_h_max] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_kernal_h] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_kernal_h_stride] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l1_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l1_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l2_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l2_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l3_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l3_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_output_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l1_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l1_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l2_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l2_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l3_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l3_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scw_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l1_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l1_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l2_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l2_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l3_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l3_condition] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_clus_2_scr_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_width_ratio] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_height_ratio] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_width] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_input_height] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_output_width] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_resize_param_output_height] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_width] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_param_input_height] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_width] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_param_output_height] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_mode] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_width] = 0
    para_buf[VpuRegister.vpu_2_para_i_clus_2_pooling_padding_height] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_ad_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_ad_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_ad_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_ad_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_ad_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_ad_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_ad_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_ad_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_ad_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_ad_3] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_ad_3] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_ad_3] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_mode_enable] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scr_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_condit1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_condit0] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_2] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_block_scw_ad_jump_1] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_line_buffer_w_max] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_line_buffer_h_max] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_kernal_h] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_kernal_h_stride] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l1_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l1_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l2_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l2_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l3_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l3_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_output_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l1_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l1_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l2_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l2_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l3_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l3_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scw_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l1_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l1_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l2_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l2_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l3_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l3_condition] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l1_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l2_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_clus_3_scr_l3_addr_step] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_width_ratio] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_height_ratio] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_width] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_input_height] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_output_width] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_resize_param_output_height] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_width] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_param_input_height] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_width] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_param_output_height] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_mode] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_width] = 0
    para_buf[VpuRegister.vpu_3_para_i_clus_3_pooling_padding_height] = 0
    para_buf = get_vpu_dict(para_buf)
    vpu_dict = vpu_param(para_buf)

    path = f'/data/xieshuhan/tinynpu/test/model/yolov3/2/block0/vpu_param.txt'
    vpu_writeTxt(vpu_dict, ADDR, path)
