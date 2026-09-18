# source: most of the llm code was taken/modified from https://github.com/paulilioaica/Llama2-Pytorch
#         neurosim - https://github.com/neurosim/DNN_NeuroSim_V2.1

from llama import Llama2
import torch
import argparse
import pickle
import argparse
from utee import misc
import os
from utee import make_path
from datetime import datetime

with open("model.pkl", "rb") as file:
    weights = pickle.load(file)

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', default=2)
parser.add_argument('--wl_grad', default=8)
parser.add_argument('--wl_activate', default=8)
parser.add_argument('--wl_error', default=8)
parser.add_argument('--inference', default=1)
parser.add_argument('--onoffratio', default=10)
parser.add_argument('--cellBit', default=1)
parser.add_argument('--subArray', default=128)
parser.add_argument('--ADCprecision', default=5)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)

# XXX Algorithm Parameters
args = parser.parse_args()
args.num_layers     = 6
args.num_hidden     = 288
args.num_ffn_hidden = 768 
args.n_heads        = 6
args.num_kv_heads   = 6
args.seq_len        = 256
args.vocab_size     = 32000
args.inference = 1            # set to run inference simulation

# XXX Hardware Properties
args.subArray = 128           # size of subArray (e.g. 128*128)
args.ADCprecision = 128         # ADC precision (e.g. 5-bit)
args.cellBit = 128              # cell precision (e.g. 4-bit/cell)
args.onoffratio = 10          # device on/off ratio (e.g. Gmax/Gmin = 3)

# XXX if do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
args.vari = 0.0                 # conductance variation (e.g. 0.1 standard deviation to generate random variation)
args.t = 0                    # retention time
args.v = 0                    # drift coefficient
args.detect = 1               # if 1, fixed-direction drift, if 0, random drift
args.target = 0.5             # drift target for fixed-direction drift

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")

num_layers     = 6
num_hidden     = 288
num_ffn_hidden = 768 
n_heads        = 6
num_kv_heads   = 6
seq_len        = 256
vocab_size     = 32000

model = Llama2(num_layers, num_hidden, num_ffn_hidden, n_heads, num_kv_heads, seq_len, vocab_size, weights, logger, args)

x = torch.randint(0, vocab_size, (1, seq_len))

output = model(x)

print(" --- Hardware Properties --- ")
print("subArray size: ")
print(args.subArray)
print("ADC precision: ")
print(args.ADCprecision)
print("cell precision: ")
print(args.cellBit)
print("on/off ratio: ")
print(args.onoffratio)
print("variation: ")
print(args.vari)

print(output)

finish_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
