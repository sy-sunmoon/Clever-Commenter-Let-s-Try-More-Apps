import torch
import torch.utils.data
from torch.nn.init import xavier_uniform_
import lr_scheduler as L

import os
import argparse
import pickle
import time
from collections import OrderedDict

import opts
import models
import utils
import codecs
import random
import numpy as np

parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opts.convert_to_config(opt, config)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
	torch.cuda.set_device(opt.gpus[0])
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True


def load_data():
	print('loading data...\n')
	data = pickle.load(open(config.data+'data.pkl', 'rb'))
	testset = utils.BiDataset(data['test'], char=config.char)
	testloader = torch.utils.data.DataLoader(dataset=testset,
											 batch_size=config.batch_size,
											 shuffle=False,
											 num_workers=0,
											 collate_fn=utils.padding)
	src_vocab = data['dict']['src']
	tgt_vocab = data['dict']['tgt']
	config.src_vocab_size = src_vocab.size()
	config.tgt_vocab_size = tgt_vocab.size()

	return {'testset': testset, 'testloader': testloader,
			'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}


def build_model(checkpoints, print_log):
	for k, v in config.items():
		print_log("%s:\t%s\n" % (str(k), str(v)))

	# model
	print('building model...\n')
	model = getattr(models, config.model)(config, src_padding_idx=utils.PAD, tgt_padding_idx=utils.PAD)
	if checkpoints is not None:
		model.load_state_dict(checkpoints['model'])
	if use_cuda:
		model.cuda()

	# print log
	param_count = 0
	for param in model.parameters():
		param_count += param.view(-1).size()[0]
	for k, v in config.items():
		print_log("%s:\t%s\n" % (str(k), str(v)))
	print_log("\n")
	print_log(repr(model) + "\n\n")
	print_log('total number of parameters: %d\n\n' % param_count)

	return model, print_log


def test_model(model, data, params):

	model.eval()
	reference, candidate, source, alignments = [], [], [], []
	count, total_count = 0, len(data['testset'])
	testloader = data['testloader']
	tgt_vocab = data['tgt_vocab']

	for src1, src2, tgt, src_len1, src_len2, tgt_len, original_src1, original_src2, original_tgt in testloader:

		if config.use_cuda:
			src1 = src1.cuda()
			src2 = src2.cuda()
			src_len1 = src_len1.cuda()
			src_len2 = src_len2.cuda()

		with torch.no_grad():
			samples, alignment = model.sample(src1, src_len1, src2, src_len2)

		candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
		source += original_src1
		reference += original_tgt
		if alignment is not None:
			alignments += [align for align in alignment]

		count += len(original_src1)
		utils.progress_bar(count, total_count)

	if config.unk and config.attention != 'None':
		cands = []
		for s, c, align in zip(source, candidate, alignments):
			cand = []
			for word, idx in zip(c, align):
				if word == utils.UNK_WORD and idx < len(s):
					try:
						cand.append(s[idx])
					except:
						cand.append(word)
						print("%d %d\n" % (len(s), idx))
				else:
					cand.append(word)
			cands.append(cand)
			if len(cand) == 0:
				print('Error!')
		candidate = cands

	with codecs.open(params['log_path']+'candidate.txt', 'w+', 'utf-8') as f:
		for i in range(len(candidate)):
			f.write(" ".join(candidate[i])+'\n')

	score = {}
	for metric in config.metrics:
		score[metric] = getattr(utils, metric)(
			reference, candidate, params['log_path'], params['log'], config)

	return score


def build_log():

	if not os.path.exists(config.logF):
		os.makedirs(config.logF)
	if opt.log == '':
		log_path = config.logF + str(int(time.time() * 1000)) + '/'
	else:
		log_path = config.logF + opt.log + '/'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	print_log = utils.print_log(log_path + 'log.txt')
	return print_log, log_path


def main():
	
	assert opt.restore
	print('loading checkpoint...\n')
	checkpoints = torch.load(opt.restore)

	# data: dict, {'trainset', 'validset', 'traindloader', 'validloader', 'src_vocab', 'tgt_vocab'}
	data = load_data()
	print_log, log_path = build_log()
	model, print_log = build_model(checkpoints, print_log)
	params = {'log_path': log_path, 'log': print_log}
	for metric in config.metrics:
		params[metric] = [] 

	score = test_model(model, data, params)


if __name__ == '__main__':
	main()
