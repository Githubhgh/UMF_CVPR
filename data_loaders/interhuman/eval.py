import sys
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch

from datetime import datetime

from .utils.metrics import *
from .datasets.evaluator import *
from collections import OrderedDict
#from utils.plot_script import *
from .utils.utils import *

from os.path import join as pjoin
from tqdm import tqdm
import json
import copy

from .interhuman import *
from umf.data.utils import ih_collate
def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt.MODE)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True, collate_fn=ih_collate)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset



def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader





os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
# torch.multiprocessing.set_sharing_strategy('file_system')

#############################################
# self-defined function
def get_time():
    from datetime import datetime

    # Get the current date and time
    current_datetime = datetime.now()

    # Format the current date and time as a string
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    datetime_string.replace(":", "_")
    return datetime_string


def failure_case_calculation(batch, dist_mat):
    diag_dist_mat = np.diagonal(dist_mat)
    top_three_indices = np.argsort(diag_dist_mat)[-3:][::-1]

    failure_case_0 = [batch[0][top_three_indices[0]], diag_dist_mat[top_three_indices[0]]]
    failure_case_1 = [batch[0][top_three_indices[1]], diag_dist_mat[top_three_indices[1]]]
    failure_case_2 = [batch[0][top_three_indices[2]], diag_dist_mat[top_three_indices[2]]]


    return failure_case_0, failure_case_1, failure_case_2


def find_max_error_indices(A, B, num_indices=3):
    #找出两个数组中误差最大的子数组序号列表。
    # 确定数组的大小
    n = A.shape[0]
    
    # 计算每个子数组之间的欧氏距离
    errors = np.zeros((n, n))
    print('Finding the failure case >>>>> May take some while')
    for i in tqdm(range(n)):
        for j in range(n):
            errors[i, j] = np.linalg.norm(A[i] - B[j])
    
    # 找出误差最大的子数组的序号
    #flat_errors = errors.flatten()
    flat_errors = np.diagonal(errors)
    max_error_indices = np.argsort(flat_errors)[-num_indices:]
    
    return max_error_indices



import re

def parse_exsiting_log_file(log_file):
    all_metrics = OrderedDict({
        'MM Distance': OrderedDict(),
        'R_precision': OrderedDict(),
        'FID': OrderedDict(),
        'Diversity': OrderedDict(),
        'MultiModality': OrderedDict()
    })

    with open(log_file, 'r') as file:
        current_replication = None
        for line in file:
            current_key = None
            replication_match = re.match(r"==================== Replication (\d+)", line)
            if replication_match:
                current_replication = int(replication_match.group(1))
            elif "---> [ground truth]" in line:
                current_key = 'ground truth'
            elif "---> [InterGen]" in line:
                current_key = 'InterGen'
            if current_key:
                if "MM Distance:" in line:
                    match = re.search(r"MM Distance: (\d+\.\d+)", line)
                    if match:
                        try:
                            all_metrics['MM Distance'][current_key] += [float(match.group(1))]
                        except KeyError:
                            all_metrics['MM Distance'][current_key] = [float(match.group(1))]
                elif "R_precision:" in line:
                    match = re.search(r"R_precision: \(top 1\): (\d+\.\d+) \(top 2\): (\d+\.\d+) \(top 3\): (\d+\.\d+)", line)
                    if match:
                        try:
                            all_metrics['R_precision'][current_key] += [np.array([float(match.group(1)),
                                                                        float(match.group(2)),
                                                                        float(match.group(3))])]
                        except KeyError:
                            all_metrics['R_precision'][current_key] = [np.array([float(match.group(1)),
                                                                       float(match.group(2)),
                                                                       float(match.group(3))])]
                elif "FID:" in line:
                    match = re.search(r"FID: (\d+\.\d+)", line)
                    if match:
                        try:
                            all_metrics['FID'][current_key] += [float(match.group(1))]
                        except KeyError:
                            all_metrics['FID'][current_key] = [float(match.group(1))]
                elif "Diversity:" in line:
                    match = re.search(r"Diversity: (\d+\.\d+)", line)
                    if match:
                        try:
                            all_metrics['Diversity'][current_key] += [float(match.group(1))]
                        except KeyError:
                            all_metrics['Diversity'][current_key] = [float(match.group(1))]
                elif "Multimodality:" in line:
                    match = re.search(r"Multimodality: (\d+\.\d+)", line)
                    if match:
                        try:
                            all_metrics['MultiModality'][current_key] += [float(match.group(1))]
                        except KeyError:
                            all_metrics['MultiModality'][current_key] = [float(match.group(1))]

    return all_metrics, current_replication

#############################################


def evaluate_matching_score(eval_wrapper, motion_loaders, file, return_failure_case = False):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        fail_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                
                # if motion_loader_name=='InterGen':
                #     fail_case = failure_case_calculation(batch, dist_mat)
                #     fail_list.append(fail_case)
                #     plot_gene_motion(batch, fail_case, name=None)
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())




            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        #print(f'---> [{motion_loader_name}] Worst Motion ID: {fail_list:.4f}')
        #print(f'---> [{motion_loader_name}] Worst Motion ID: {fail_list:.4f}', file=file, flush=True)


        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)  # -> (1056, 512) -> 1056: test size; 512: interclip embedding size
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # 尝试计算 Fréchet 距离，捕获可能的异常
        try:
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov) # (512)
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        except ValueError as e:
            # 处理包含 inf 或 NaN 的数组导致的异常
            print(f"Error calculating FID for {model_name}: {e}")
            print(f"Error calculating FID for {model_name}: {e}", file=file, flush=True)
            # 可以选择为此模型设置一个错误值或跳过此模型的 FID 计算
            eval_dict[model_name] = 999  # 或者设置一个特定的错误值
    return eval_dict


def evaluate_diversity(eval_wrapper, activation_dict, diversity_times, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        try:
            diversity = calculate_diversity(motion_embeddings, diversity_times)
            eval_dict[model_name] = diversity
            print(f'---> [{model_name}] Diversity: {diversity:.4f}')
            print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
        except ValueError as e:
            print(f"Error calculating Diversity for {model_name}: {e}")
            print(f"Error calculating Diversity for {model_name}: {e}", file=file, flush=True)
            # 可以选择为此模型设置一个错误值或跳过此模型的 FID 计算
            eval_dict[model_name] = 0  # 或者设置一个特定的错误值
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, mm_num_times, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()

            try:
                multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
            except ValueError as e:
                print(f"Error calculating multimodality for {model_name}: {e}")
                print(f"Error calculating multimodality for {model_name}: {e}", file=file, flush=True)
                multimodality = 0
        
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval



def eval_in_train(model, epoch, device, log_file, time='0', test_data = "data_loaders/interhuman/configs/datasets.yaml"):

    log_file = log_file + str(time)
    eval_at_train_save_dir = os.path.dirname(log_file)
    mm_num_samples = 100 #100
    mm_num_repeats = 30 #30
    mm_num_times = 10 #10
    
    diversity_times = 300 #300
    replication_times = 1

    # batch_size is fixed to 96!!
    batch_size = 96

    if '.yaml' in test_data:
        data_cfg = get_config(test_data).interhuman_test
    else:
        data_cfg = test_data

    cfg_path_list = ["data_loaders/interhuman/configs/model.yaml"]


    eval_motion_loaders = {}


    eval_motion_loaders['umf'] = lambda: get_motion_loader(
                                            batch_size,
                                            model,
                                            gt_dataset,
                                            device,
                                            mm_num_samples,
                                            mm_num_repeats,
                                            )

    #device = torch.device('cuda:%d' % gpu_index if torch.cuda.is_available() else 'cpu')
    
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    evalmodel_cfg = get_config("data_loaders/interhuman/configs/eval_model.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device) # CLIP embedding

    #log_file = f'./evaluation_{1}.log'
    
    
    # Extracting step number from checkpoint path
    #step_number = os.path.basename(checkpoint_path).split('=')[2].split('.')[0]

    # Creating log file if it does not exist
    #log_file = f'./evaluation_{step_number}.log'
    with open(log_file, 'a'):
        os.utime(log_file, None)

    with open(log_file, 'a') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Epoch {epoch} ====================')
            print(f'==================== Epoch {epoch} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)
            #fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f, save_dir =  eval_at_train_save_dir, epoches = epoch, train = True)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(eval_wrapper, acti_dict, diversity_times, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, mm_num_times, f)


            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)
 
    return (round(fid_score_dict['InterGen'], 3),  # FID
            round(R_precision_dict['InterGen'][0], 3), round(R_precision_dict['InterGen'][1], 3), round(R_precision_dict['InterGen'][2],3), # R_precision
            round(div_score_dict['InterGen'], 3),
            round(mm_score_dict['InterGen'], 3)
    )

