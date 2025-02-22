import argparse
import logging
import os
import random
import sys
import SimpleITK as sitk

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from networks.DAEFormer import DAEFormer
from trainer import trainer_synapse
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="./data/Synapse/",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")  #类别
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--test_path", type=str, default="data/Synapse/Gu_Lou_test_data", help="root dir for test data")
parser.add_argument("--model_pth", type=str, default="./best_model.pth", help="list dir")   #权重路径
parser.add_argument("--test_split", type=str, default="Gu_Lou_test", help="test data list")  #测试集识别标签
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", type=bool, default=True, help="whether to save results during inference")
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

args = parser.parse_args()
# if args.dataset == "Synapse":
#     args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.test_path, split=args.test_split, img_size=args.img_size, list_dir=args.list_dir, num_classes=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)
        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f" % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f" % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    net = DAEFormer(num_classes=args.num_classes).cuda(0)

    snapshot = args.model_pth
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "transfilm_epoch_" + str(args.max_epochs - 1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = snapshot.split("/")[-1]


    log_folder = "./test_log/test_log_"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

    #预测完毕开始
    result_root = r'./model_out/predictions/'
    # file_list
    result_list = os.listdir(result_root)
    for data_num in range(len(result_list)):
        result_name = result_list[data_num]
        if result_name.endswith('.nii.gz') and result_name.split('_')[1].split('.')[0] == 'pred':
            result_pred_path = result_root + result_name
            result_gt_path = result_root + result_name.replace('pred', 'gt')
            print(result_pred_path)
            print(result_gt_path)

            pred = sitk.ReadImage(result_pred_path)
            gt = sitk.ReadImage(result_gt_path)
            pred_context = sitk.GetArrayFromImage(pred)
            gt_context = sitk.GetArrayFromImage(gt)
            # image_context = (image_context - image_context.min()) / (image_context.max() - image_context.min()) #图像归一化
            # label_context = (label_context - label_context.min()) / (label_context.max() - label_context.min())  # 图像归一化
            (x, y, z) = pred_context.shape

            pred_total_value = 0
            gt_total_value = 0

            for slice in range(x):
                slice_pred = pred_context[slice]
                slice_gt = gt_context[slice]
                pred_value = np.count_nonzero(slice_pred == 1)
                pred_total_value += pred_value
                gt_value = np.count_nonzero(slice_gt == 1)
                gt_total_value += gt_value
                print('当前为' + result_name + '的第' + str(slice + 1) + '个切片：预测像素' + str(
                    pred_value) + '个 实际像素' + str(gt_value) + '个')

            pred_area =  pred_total_value/(x*625*6)# 预测总面积

            print('当前为' + result_name + '：总预测像素' + str(pred_total_value) + '个 总实际像素' + str(
                gt_total_value) + '个')
            print('预测总面积为：' + str(pred_area))