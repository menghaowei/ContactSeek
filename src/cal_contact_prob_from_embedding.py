# -*- coding: UTF-8 -*-

import sys
import argparse
import numpy as np
import logging
import json


# Version information START --------------------------------------------------
VERSION_INFO = """

Author: MENG Howard

Version-01:
    2025-04-19
        Calculate contact probability from AF3 embedding results

E-Mail: meng_howard@126.com

"""
# Version information END ----------------------------------------------------


########################################################################
# def function
########################################################################
def get_contact_prob_from_embedding_multi_threshold(embedding_res, 
                                                    linear_weights, 
                                                    threshold_list = [8]):
    """
    INPUT:
        <embedding_res> np.array  n*n*c

        <linear_weights> np.array c*m

    RETURN 
        <contact_prob> np.array n*n
    """
    # AF3 defaults 
    first_break = 2.3125
    last_break = 21.6875
    num_bins = 64
    _CONTACT_EPSILON = 1e-3
    
    # 1. 计算对称 logits
    left_half_logits = np.dot(embedding_res, linear_weights)           # [1557, 1557, 64]

    logits = left_half_logits + np.swapaxes(left_half_logits, -2, -3)  # 对称相加

    # 2. 计算概率分布
    max_logits = np.max(logits, axis=-1, keepdims=True)          # 沿分箱维度取最大值 [1557, 1557, 1]
    stable_logits = logits - max_logits                          # 减去最大值避免溢出
    exp_logits = np.exp(stable_logits)                           # 安全计算指数
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # 归一化概率

    ## 3. 生成分箱边界
    breaks = np.linspace(first_break, last_break, num_bins - 1)  # 63个分界点
    bin_tops = np.append(breaks, breaks[-1] + (breaks[-1] - breaks[-2]))  # 64个分箱上界

    # 4. 计算接触概率
    res_list = []
    
    for run_threshold in threshold_list:
        threshold_up = run_threshold + _CONTACT_EPSILON
        is_contact_bin = (bin_tops <= threshold_up).astype(np.float32)
        contact_probs = np.einsum('ijk,k->ij', probs, is_contact_bin)  # [1557, 1557]
        res_list.append(contact_probs)

    return res_list


########################################################################
# calculation part
########################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Contact probability from AF3 embedding file")

    parser.add_argument("--af3_json",
                        help="confidences.json from AF3 output", required=True)

    parser.add_argument("--af3_embedding",
                        help="embeddings.npz from AF3 output", required=True)
    
    parser.add_argument("--af3_weigth",
                        help="AF3 model weigths for contact prob.", required=True)

    parser.add_argument("--threshold_list",
                        help="thresholds of contact probability, defalut = 8", 
                        type=str, 
                        default="8")

    parser.add_argument("--output", 
                        help="file name of .npz", 
                        default="af3_contact_prob.npz")

    ARGS, unknown_args = parser.parse_known_args()
    # ---------------------------------------------------------------->>>>>>>>
    # load params
    # ---------------------------------------------------------------->>>>>>>>
    logging.basicConfig(level=(4 - 3) * 10,
                    format='%(levelname)-5s @ %(asctime)s: %(message)s ',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stderr,
                    filemode="w")

    logging.info("Start" )
    logging.info(ARGS.af3_json)
    logging.info(ARGS.af3_embedding)
    logging.info(ARGS.af3_weigth)
    logging.info("set threshold list: ")
    logging.info(str(ARGS.threshold_list))
    
    query_threshold_list = list( map(int, str(ARGS.threshold_list).split(",")) )
    logging.info("run threshold list: " + str(query_threshold_list))

    # ---------------------------------------------------------------->>>>>>>>
    # run part
    # ---------------------------------------------------------------->>>>>>>>
    # 加载保存的 .npz 文件
    loaded = np.load(ARGS.af3_weigth, allow_pickle=True)
    linear_weights = loaded['weights']

    # load json count token num
    with open(ARGS.af3_json, "r") as af3_contact_file:
        af_json = json.load(af3_contact_file)

    token_num = len(af_json["token_res_ids"])

    # load embedding
    load_embeddings = {}
    with open(ARGS.af3_embedding, "rb") as f:
        embeddings_obj = np.load(f)
        logging.info("loading single...")
        load_embeddings["single_embeddings"] = embeddings_obj["single_embeddings"][:token_num]
        logging.info("loading pair...")
        load_embeddings["pair_embeddings"] = embeddings_obj["pair_embeddings"][:token_num, :token_num]
        logging.info("loading embedding results.... Done!")

    # 计算接触概率
    contact_prob_list = get_contact_prob_from_embedding_multi_threshold(
        load_embeddings["pair_embeddings"], 
        linear_weights, 
        query_threshold_list
    )

    # save contact prob
    np.savez(ARGS.output, contact_res = np.array(contact_prob_list))

    logging.info("output: " + ARGS.output)
    logging.info("Done!" )

