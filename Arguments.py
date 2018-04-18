import argparse

# ARGUMENT PART
parser = argparse.ArgumentParser(description='Seq2Seq NMT')
# option
parser.add_argument('-no_gpu', type=bool, default=False, help='disable the gpu')
# model
parser.add_argument('-hidden_size', type=int, default=256)
parser.add_argument('-embed_size', type=int, default=128)
parser.add_argument('-max_sent', type=int, default=10, help='max sentence length')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-model', type=str, default='vanilla')
parser.add_argument('-model_name', type=str, default='NONE')
parser.add_argument('-enc_unit', type=str, default='syll')
parser.add_argument('-dec_unit', type=str, default='syll')
# learning
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=1, help='batch size for training [default: 64]')
parser.add_argument('-learning_rate', type=float, default=0.02, help='learning rate')
parser.add_argument('-kfold', type=int, default=10, help='k-folding size')
parser.add_argument('-early', type=int, default=None)
parser.add_argument('-optim', type=str, default='SGD')
parser.add_argument('-exam_unit', type=str, default='word')
# Data
parser.add_argument('-train_size', type=int, default=3000, help='train size')
parser.add_argument('-task', type=str, default='train')
parser.add_argument('-files_to_read', type=int, default=10, help='the number of files to read for test data')
args = parser.parse_args()
