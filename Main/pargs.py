import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo-2class')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=128)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tddroprate', type=float, default=0.0)
    parser.add_argument('--budroprate', type=float, default=0.0)
    parser.add_argument('--hid_feats', type=int, default=128)
    parser.add_argument('--out_feats', type=int, default=128)

    parser.add_argument('--diff_lr', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
