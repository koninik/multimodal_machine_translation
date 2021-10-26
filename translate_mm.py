''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import numpy as np

from dataset_mm import translate_collate_fn, collate_fn, TranslationDataset
from transformer_mmt.Translator import Translator
from preprocess_mm import read_instances_from_file, convert_instance_to_idx_seq

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate_mm.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-test_image_feat', required=True)
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(args.vocab)
    preprocess_settings = preprocess_data['settings']
    test_image_feat = np.load(args.test_image_feat)
    test_src_word_insts = read_instances_from_file(
        args.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            image_features=test_image_feat,
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=translate_collate_fn)

    translator = Translator(opt)

    # self._src_insts[idx], self._tgt_insts[idx], self.image_features[idx]
    with open(args.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            src_seq, src_pos, image_features = map(lambda x: x.to(translator.device), batch)
            all_hyp, all_scores = translator.translate_batch(src_seq, src_pos, image_features)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line + '\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
