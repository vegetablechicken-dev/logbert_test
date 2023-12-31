import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import torch
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from bert_pytorch.dataset.sample import fixed_window


def compute_anomaly(results, params, seq_threshold):
    is_logkey = params["is_logkey"]
    is_time = params["is_time"]
    total_errors = 0
    # 建议根据情况修改
    abnormal_loss_bond, abnormal_prob_bond = 0.95, 0.05
    for key, value in results[0]:
        if results[1][key][0] > abnormal_loss_bond or results[1][key][1] < abnormal_prob_bond:
            total_error += 1
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, params, th_range, seq_range):
    best_result = [0] * 9
    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results, params, seq_th)
        TP = compute_anomaly(test_abnormal_results, params, seq_th)
        
        if TP == 0:
            continue

        TN = len(test_normal_results[0]) - FP
        FN = len(test_abnormal_results[0]) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print("seq_th:",seq_th)
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
        if F1 > best_result[-1]:
            best_result = [0, seq_th, FP, TP, TN, FN, P, R, F1]
    return best_result


class Predictor():
    def __init__(self, options):
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.device = options["device"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.seq_len = options["seq_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.num_candidates = options["num_candidates"]
        self.output_dir = options["output_dir"]
        self.model_dir = options["model_dir"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale_path = options["scale_path"]

        self.hypersphere_loss = options["hypersphere_loss"]
        self.hypersphere_loss_test = options["hypersphere_loss_test"]

        self.lower_bound = self.gaussian_mean - 3 * self.gaussian_std
        self.upper_bound = self.gaussian_mean + 3 * self.gaussian_std

        self.center = None
        self.radius = None
        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len=options["min_len"]
        self.key_dict = {}
        self.seqs_to_keys = {}
        self.seqs_dict_idx = 0

        self.abnormal_loss_bond = 0.95
        self.abnormal_prob_bond = 0.05
    '''
    修改此处
    original method(logbert): 如果不在候选集中前g个(num_candidates),则为异常
    change:
    '''
    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        output_maskes = []
        '''
        for i, token in enumerate(masked_label):
            # output_maskes.append(torch.argsort(-masked_output[i])[:30].cpu().numpy()) # extract top 30 candidates for mask labels

            # if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
                # num_undetected_tokens += 1
            if token not in self.key_dict:
                score_loss, score_prob = 0, 0
                for 
        '''
        return num_undetected_tokens, [output_maskes, masked_label.cpu().numpy()]

    @staticmethod
    def generate_test(output_dir, file_name, window_size, adaptive_window, seq_len, scale, min_len):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """
        log_seqs = []
        tim_seqs = []
        with open(output_dir + file_name, "r") as f:
            for idx, line in tqdm(enumerate(f.readlines())):
                # if idx > 12000: break
                log_seq, tim_seq = fixed_window(line, window_size,
                                                adaptive_window=adaptive_window,
                                                seq_len=seq_len, min_len=min_len)
                if len(log_seq) == 0:
                    continue

                # if scale is not None:
                #     times = tim_seq
                #     for i, tn in enumerate(times):
                #         tn = np.array(tn).reshape(-1, 1)
                #         times[i] = scale.transform(tn).reshape(-1).tolist()
                #     tim_seq = times

                log_seqs += log_seq
                tim_seqs += tim_seq

        # sort seq_pairs by seq len
        log_seqs = np.array(log_seqs)
        tim_seqs = np.array(tim_seqs)

        test_len = list(map(len, log_seqs))
        test_sort_index = np.argsort(-1 * np.array(test_len))

        log_seqs = log_seqs[test_sort_index]
        tim_seqs = tim_seqs[test_sort_index]

        print(f"{file_name} size: {len(log_seqs)}")
        return log_seqs, tim_seqs

    def helper(self, model, output_dir, file_name, vocab, scale=None, error_dict=None):
        key_dict = {}
        seqs_to_keys = {}
        seqs_dict_idx = 0
        total_results = []
        total_errors = []
        output_results = []
        total_dist = []
        output_cls = []
        logkey_test, time_test = self.generate_test(output_dir, file_name, self.window_size, self.adaptive_window, self.seq_len, scale, self.min_len)

        # use 1/10 test data
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test * 0.1)] if isinstance(self.test_ratio, float) else rand_index[:self.test_ratio]
            print(rand_index.shape)
            logkey_test, time_test = logkey_test[rand_index], time_test[rand_index]


        seq_dataset = LogDataset(logkey_test, time_test, vocab, seq_len=self.seq_len,
                                 corpus_lines=self.corpus_lines, on_memory=self.on_memory, predict_mode=True, mask_ratio=self.mask_ratio)

        # use large batch size in test data
        data_loader = DataLoader(seq_dataset, batch_size=1, num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)
        idx = 0
        for idx, data in tqdm(enumerate(data_loader)):
            idx += 1
            if idx > 100:
                break
            # data = {key: value.to(self.device) for key, value in data.items()}
            data_dict = {}
            log_input = data["bert_input"]
            time_input = data["time_input"]
            
            if log_input in seqs_to_keys.values():
                key_of_seqs = seqs_to_keys.get(log_input)
                log_loss, log_prob = key_dict[key_of_seqs][0], key_dict[key_of_seqs][1]
            else:
                score_loss, score_prob = [], []
                # log_input.to(device)
                # time_input.to(device)
                for i in range(1, len(log_input)):
                    log_label, log_input[0, i] = log_input[0, i], 4
                    time_label, time_input[0, i] = time_input[0, i], 0
                    result = model(log_input, time_input)
                    mask_lm_output, mask_tm_output = result["logkey_output"], result["time_output"]
                    prob, loss = mask_lm_output[0, log_label], -1 * log(mask_lm_output[0, log_label])
                    score_loss.append(loss)
                    score_prob.append(prob)
                score_loss, _ = torch.sort(torch.tensor(score_loss).float(), descending=True)
                score_prob, _ = torch.sort(torch.tensor(score_prob).float(), descending=True)
                abnormal_loss, abnormal_prob = torch.mean(score_loss[:self.num_candidates]).item(), torch.mean(score_prob[:self.num_candidates]).item()
                seqs_to_keys[seqs_dict_idx] = log_input
                key_dict[seqs_dict_idx] = [abnormal_loss, abnormal_prob]
                log_loss, log_prob = abnormal_loss, abnormal_prob
                seqs_dict_idx += 1
            output_results = []

        return [self.seqs_to_keys, self.key_dict], output_results

        # for time
        # return total_results, total_errors

        #for logkey
    

        # for hypersphere distance
        # return total_results, output_cls

    def predict(self):
        # 加载模型
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))

        start_time = time.time()
        vocab = WordVocab.load_vocab(self.vocab_path)

        scale = None
        error_dict = None
        if self.is_time:
            with open(self.scale_path, "rb") as f:
                scale = pickle.load(f)

            with open(self.model_dir + "error_dict.pkl", 'rb') as f:
                error_dict = pickle.load(f)

        if self.hypersphere_loss:
            center_dict = torch.load(self.model_dir + "best_center.pt")
            self.center = center_dict["center"]
            self.radius = center_dict["radius"]
            # self.center = self.center.view(1,-1)


        print("test normal predicting")
        test_normal_results, test_normal_errors = self.helper(model, self.output_dir, "log_normal", vocab, scale, error_dict)
        
        print("#############")
        print(test_normal_results)
        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_errors = self.helper(model, self.output_dir, "log_anormaly", vocab, scale, error_dict)
        
        print("##############")
        print(test_abnormal_results)
        print("Saving test normal results")
        with open(self.model_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        print("Saving test normal errors")
        with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
            pickle.dump(test_normal_errors, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
            pickle.dump(test_abnormal_errors, f)

        params = {"is_logkey": self.is_logkey, "is_time": self.is_time, "hypersphere_loss": self.hypersphere_loss,
                  "hypersphere_loss_test": self.hypersphere_loss_test}
        best_th, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(test_normal_results,
                                                                            test_abnormal_results,
                                                                            params=params,
                                                                            th_range=np.arange(10),
                                                                            seq_range=np.arange(0,1,0.1))

        print("best threshold: {}, best threshold ratio: {}".format(best_th, best_seq_th))
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

