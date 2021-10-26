''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn

from dataset_mm import translate_collate_fn, collate_fn, TranslationDataset
from transformer_mmt_emb_img.Translator import Translator #missing
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

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

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_image_feat = np.load(opt.test_image_feat)
    test_src_word_insts = read_instances_from_file(
        opt.src,
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
        batch_size=opt.batch_size,
        collate_fn=translate_collate_fn)

    translator = Translator(opt)

    prediction = []
    # self._src_insts[idx], self._tgt_insts[idx], self.image_features[idx]
    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            src_seq, src_pos, image_features = map(lambda x: x.to(translator.device), batch)
            all_hyp, all_scores = translator.translate_batch(src_seq, src_pos, image_features)
            
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    prediction += [pred_line]
                    #for layer in range(2):
                     #   print(translator.model.encoder.layer_stack[layer].slf_attn.attention.alphas.shape)
                    #src = ' '.join([test_loader.dataset.src_idx2word[idx] + ["</s>"] for idx in idx_seq])
                    f.write(pred_line + '\n')
    print('[Info] Finished.')

    #idx = 5
    tgt_sent = prediction[208].split()
    print(tgt_sent)  
    #src_sent = str(test_src_word_insts[209])
    #src_sent = 'two construction workers have a discussion while on the work site .'.split() #209
    src_sent = 'a worker in an orange vest is using a shovel .'.split()
    print(src_sent)
    
   # model = opt.model

    def draw(data, x, y, ax):
        seaborn.heatmap(data.cpu(), 
                    xticklabels=x, square=True, yticklabels=y, 
                    cbar=False, ax=ax)


    def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
        
        image = Image.open(image_path)
        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

        words = [rev_word_map[ind] for ind in seq]

        for t in range(len(words)):
            if t > 50:
                break
            plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[t, :]
            if smooth:
                alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
            else:
                alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        plt.show()
    
    for layer in range(3):
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        
        print("Encoder Layer", layer+1)
        
        for h in range(4):
            
            draw(translator.model.encoder.layer_stack[layer].slf_attn.attention.alphas[h].data[:len(src_sent), :len(src_sent)],
                src_sent, src_sent if h ==0 else [], ax=axs[h])
            #plt.show()
            plt.savefig('Enc_self_layer'+ str(layer+1) + '.png')
                

    for layer in range(3):
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(translator.model.decoder.layer_stack[layer].slf_attn.attention.alphas[h].data[:len(tgt_sent), :len(tgt_sent)], 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
            #plt.show()
            2537596840.jpg
            plt.savefig('Dec_self_layer'+ str(layer+1) + '_head_1_4 ' + '.png')

        for h in range(4):
            draw(translator.model.decoder.layer_stack[layer].slf_attn.attention.alphas[h].data[:len(tgt_sent), :len(tgt_sent)], 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
            #plt.show()
            
            plt.savefig('Dec_self_layer'+ str(layer+1) + '_head_1_4 ' + '.png')

        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        for h in range(4):
            
            #plt.show()
            plt.savefig('Dec_src_layer'+ str(layer+1) + '_head_1_4' + '.png')

if __name__ == "__main__":
    main()
