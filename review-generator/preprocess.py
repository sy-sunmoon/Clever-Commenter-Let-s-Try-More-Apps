import argparse
import utils
import pickle

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('--load_data', required=True,
                    help="input file for the data")
parser.add_argument('--save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('--src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('--tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('--src_char', action='store_true',
                    help='character based encoding')
parser.add_argument('-tgt_char', action='store_true',
                    help='character based decoding')
parser.add_argument('--src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('-tgt_suf', default='tgt',
                    help="the suffix of the target filename")
parser.add_argument('--lower', action='store_true',
                    help='lower the case')
parser.add_argument('--share', action='store_true',
                    help='share the vocabulary between source and target')
parser.add_argument('--freq', type=int, default=0,
                    help="remove words less frequent")

parser.add_argument('--report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, char, vocab, size, freq=0):
	
    max_length = 0
    with open(filename, encoding='utf8') as f:
        for sent in f.readlines():
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            max_length = max(max_length, len(tokens))
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))
    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size, freq)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile1, srcFile2, tgtFile, srcDicts, tgtDicts, save_srcFile1, save_srcFile2, save_tgtFile, trun=True):

    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s & %s ...' % (srcFile1, srcFile2, tgtFile))
    srcF1 = open(srcFile1, encoding='utf8')
    srcF2 = open(srcFile2, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')

    srcIdF1 = open(save_srcFile1 + '.id', 'w')
    srcIdF2 = open(save_srcFile2 + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF1 = open(save_srcFile1 + '.str', 'w', encoding='utf8')
    srcStrF2 = open(save_srcFile2 + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')

    while True:
        sline1 = srcF1.readline()
        sline2 = srcF2.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline1 == "" and sline2 == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline1 == "" or sline2 == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline1 = sline1.strip()
        sline2 = sline2.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline1 == "" or sline2 == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        if opt.lower:
            sline1 = sline1.lower()
            sline2 = sline2.lower()
            tline = tline.lower()

        srcWords1 = sline1.split() if not opt.src_char else list(sline1)
        srcWords2 = sline2.split() if not opt.src_char else list(sline2)
        tgtWords = tline.split() if not opt.tgt_char else list(tline)

        srcIds1 = srcDicts.convertToIdx(srcWords1, utils.UNK_WORD)
        srcIds2 = srcDicts.convertToIdx(srcWords2, utils.UNK_WORD)
        tgtIds = tgtDicts.convertToIdx(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD) 

        srcIdF1.write(" ".join(list(map(str, srcIds1)))+'\n')
        srcIdF2.write(" ".join(list(map(str, srcIds2)))+'\n')
        tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')

        if not opt.src_char:
            srcStrF1.write(" ".join(srcWords1)+'\n')
            srcStrF2.write(" ".join(srcWords2)+'\n')
        else:
            srcStrF1.write("".join(srcWords1) + '\n')
            srcStrF2.write("".join(srcWords2) + '\n')
        if not opt.tgt_char:
            tgtStrF.write(" ".join(tgtWords)+'\n')
        else:
            tgtStrF.write("".join(tgtWords) + '\n')

        sizes += 1
        count += 1
        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF1.close()
    srcF2.close()
    tgtF.close()
    srcStrF1.close()
    srcStrF2.close()
    tgtStrF.close()
    srcIdF1.close()
    srcIdF2.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF1': save_srcFile1 + '.id', 'srcF2': save_srcFile2 + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF1': save_srcFile1 + '.str', 'original_srcF2': save_srcFile2 + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}


def main():

    dicts = {}

    train_src1, train_src2, train_tgt = opt.load_data + 'train.' + \
        opt.src_suf + '1', opt.load_data + 'train.' + \
        opt.src_suf + '2',opt.load_data + 'train.' + opt.tgt_suf
    valid_src1, valid_src2, valid_tgt = opt.load_data + 'valid.' + \
        opt.src_suf + '1', opt.load_data + 'valid.' + \
        opt.src_suf + '2',opt.load_data + 'valid.' + opt.tgt_suf
    test_src1, test_src2, test_tgt = opt.load_data + 'test.' + \
        opt.src_suf + '1', opt.load_data + 'test.' + \
        opt.src_suf + '2',opt.load_data + 'test.' + opt.tgt_suf

    save_train_src1, save_train_src2, save_train_tgt = opt.save_data + 'train.' + \
        opt.src_suf + '1', opt.save_data + 'train.' + \
        opt.src_suf + '2', opt.save_data + 'train.' + opt.tgt_suf
    save_valid_src1, save_valid_src2, save_valid_tgt = opt.save_data + 'valid.' + \
        opt.src_suf + '1', opt.save_data + 'valid.' + \
        opt.src_suf + '2', opt.save_data + 'valid.' + opt.tgt_suf
    save_test_src1, save_test_src2, save_test_tgt = opt.save_data + 'test.' + \
        opt.src_suf + '1', opt.save_data + 'test.' + \
        opt.src_suf + '2', opt.save_data + 'test.' + opt.tgt_suf

    src_dict, tgt_dict = opt.save_data + 'src.dict', opt.save_data + 'tgt.dict'

    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('Building source and target vocabulary...')
        dicts['src'] = dicts['tgt'] = utils.Dict(
            [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
        dicts['src'] = makeVocabulary(
            train_src1, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = makeVocabulary(
            train_src2, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = dicts['tgt'] = makeVocabulary(
            train_tgt, opt.tgt_char, dicts['src'], opt.tgt_vocab_size, freq=opt.freq)
    else:
        print('Building source vocabulary...')
        dicts['src'] = utils.Dict(
            [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
        dicts['src'] = makeVocabulary(
            train_src1, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        dicts['src'] = makeVocabulary(
            train_src2, opt.src_char, dicts['src'], opt.src_vocab_size, freq=opt.freq)
        print('Building target vocabulary...')
        dicts['tgt'] = utils.Dict(
            [utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=opt.lower)
        dicts['tgt'] = makeVocabulary(
            train_tgt, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size, freq=opt.freq)

    print('Preparing training ...')
    train = makeData(train_src1, train_src2, train_tgt,
                     dicts['src'], dicts['tgt'], save_train_src1, save_train_src2, save_train_tgt)

    print('Preparing validation ...')
    valid = makeData(valid_src1, valid_src2, valid_tgt,
                     dicts['src'], dicts['tgt'], save_valid_src1, save_valid_src2, save_valid_tgt, trun=False)

    print('Preparing test ...')
    test = makeData(test_src1, test_src2, test_tgt,
                    dicts['src'], dicts['tgt'], save_test_src1, save_test_src2, save_test_tgt, trun=False)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {'train': train, 'valid': valid,
            'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb')) 


if __name__ == "__main__":
    main()
