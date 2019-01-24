import torch
import torch.nn as nn
import utils
import models
import random


class seq2seq(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0):

        super(seq2seq, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder1 = models.rnn_encoder(config, padding_idx=src_padding_idx)
            self.encoder2 = models.rnn_encoder(config, padding_idx=src_padding_idx)

        tgt_embedding = self.encoder1.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.rnn_decoder(config, embedding=tgt_embedding, \
                                              use_attention=use_attention, padding_idx=tgt_padding_idx)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
        if config.use_cuda:
            self.criterion.cuda()

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def forward(self, src1, src_len1, src2, src_len2, dec, targets, teacher_ratio=1.0):

        lengths1, indices1 = torch.sort(src_len1, dim=0, descending=True)
        _, reverse_indices1 = torch.sort(indices1)
        src1 = torch.index_select(src1, dim=0, index=indices1)
        src1 = src1.t()

        lengths2, indices2 = torch.sort(src_len2, dim=0, descending=True)
        _, reverse_indices2 = torch.sort(indices2)
        src2 = torch.index_select(src2, dim=0, index=indices2)
        src2 = src2.t()

        dec = dec.t() # [max_tgt_len, batch_size]
        targets = targets.t()

        contexts1, state1 = self.encoder1(src1, lengths1.tolist())
        contexts2, state2 = self.encoder2(src2, lengths2.tolist())

        contexts1 = torch.index_select(contexts1, dim=1, index=reverse_indices1)
        contexts2 = torch.index_select(contexts2, dim=1, index=reverse_indices2)
        state1 = (torch.index_select(state1[0], dim=1, index=reverse_indices1),\
                  torch.index_select(state1[1], dim=1, index=reverse_indices1))
        state = state1

        if self.decoder.attention1 is not None:
            self.decoder.attention1.init_context(context=contexts1)
        if self.decoder.attention2 is not None:
            self.decoder.attention2.init_context(context=contexts2)

        outputs = []
        for input in dec.split(1):
            output, state, attn_weights = self.decoder(input.squeeze(0), state)
            outputs.append(output)
        outputs = torch.stack(outputs) 

        loss = self.compute_loss(outputs, targets)
        return loss, outputs

    def sample(self, src1, src_len1, src2, src_len2):

        lengths1, indices1 = torch.sort(src_len1, dim=0, descending=True)
        _, reverse_indices1 = torch.sort(indices1)
        src1 = torch.index_select(src1, dim=0, index=indices1)
        src1 = src1.t()

        lengths2, indices2 = torch.sort(src_len2, dim=0, descending=True)
        _, reverse_indices2 = torch.sort(indices2)
        src2 = torch.index_select(src2, dim=0, index=indices2)
        src2 = src2.t()

        bos = torch.ones(src1.size(1)).long().fill_(utils.BOS)
        if self.use_cuda:
            bos = bos.cuda()

        contexts1, state1 = self.encoder1(src1, lengths1.tolist())
        contexts2, state2 = self.encoder2(src2, lengths2.tolist())

        contexts1 = torch.index_select(contexts1, dim=1, index=reverse_indices1)
        contexts2 = torch.index_select(contexts2, dim=1, index=reverse_indices2)
        state1 = (torch.index_select(state1[0], dim=1, index=reverse_indices1),\
                  torch.index_select(state1[1], dim=1, index=reverse_indices1))
        state = state1

        if self.decoder.attention1 is not None:
            self.decoder.attention1.init_context(context=contexts1)
        if self.decoder.attention2 is not None:
            self.decoder.attention2.init_context(context=contexts2)

        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        sample_ids = outputs.t()

        if self.decoder.attention1 is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = alignments.t()
        else:
            alignments = None

        return sample_ids.tolist(), alignments