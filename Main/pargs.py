import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='Weibo')
    # parser.add_argument('--unsup_dataset', type=str, default='UWeiboV1')
    # parser.add_argument('--tokenize_mode', type=str, default='jieba')


    parser.add_argument('--dataset', type=str, default='Twitter15-tfidf')
    # tfidf dataset no need
    parser.add_argument('--unsup_dataset', type=str, default='UTwitterV1')
    # nltk, naive for en; jieba, naive for cn
    parser.add_argument('--tokenize_mode', type=str, default='nltk')

    parser.add_argument('--split', type=str, default='802')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=2)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # bigcn or udgcn
    parser.add_argument('--model', type=str, default='bigcn')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tddroprate', type=float, default=0.2)
    parser.add_argument('--budroprate', type=float, default=0.2)
    parser.add_argument('--hid_feats', type=int, default=64)
    parser.add_argument('--out_feats', type=int, default=64)

    parser.add_argument('--diff_lr', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
