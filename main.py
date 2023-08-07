from argparse import ArgumentParser
import numpy as np
import random
import yaml
import torch
from training import ensure_dir, run


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='EuroSat', type=str,
                        help='database name (default: LIVE)')
    parser.add_argument('--model', default='CNNIQA', type=str,
                        help='model name (default: CNNIQA)')
    parser.add_argument('--resume', default=None, type=str,
                         help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    # parser.add_argument('--multi_gpu', action='store_true',
    #                     help='flag whether to use multiple GPUs')

    args = parser.parse_args()

    torch.manual_seed(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    config.update(config[args.database])
    config.update(config[args.model])
    
    ensure_dir('checkpoints')
    trained_model_file = 'C:/Users/win 10/Desktop/CNNIQA/CNNIQA/checkpoints/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)
    ensure_dir('results')
    save_result_file = 'C:/Users/win 10/Desktop/CNNIQA/CNNIQA/results/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id, trained_model_file, save_result_file, args.disable_gpu)
