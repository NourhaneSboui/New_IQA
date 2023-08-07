from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches
import numpy as np
import h5py, os
from Network import CNNIQAnet
from IQADataset import read_csv, default_loader
import tifffile

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test on the whole cross dataset')
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="dataset dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")
    parser.add_argument("--save_path", type=str, default='scores',
                        help="save path (default: score)")

    args = parser.parse_args()

    device = torch.device("cpu")
    model = CNNIQAnet().to(device)

    model.load_state_dict(torch.load(args.model_file))

    
    im_names, mos_scores, mos_stds = read_csv('C:/Users/win 10/Desktop/FR_IQA/IQA-optimization/Artif_MOS.csv')


    model.eval()
    
    with torch.no_grad(): 
        scores = []   
        for i in range(len(im_names)):
            im = default_loader(os.path.join(args.dataset_dir, im_names[i]))
            patches = NonOverlappingCropPatches(im, 32, 32)
            patch_scores = model(torch.stack(patches).to(device))
            score = patch_scores.mean().item()
            print(score)
            scores.append(score)
        np.save(args.save_path, scores)