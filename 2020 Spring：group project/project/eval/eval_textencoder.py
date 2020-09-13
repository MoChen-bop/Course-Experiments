import os
import sys
sys.path.append('..')
from datetime import datetime
import h5py
import seaborn as sns

import torch

from datasets.text_embedding.cub_dataset import CUBTextEmbeddingDataset as CUBDataset 
from datasets.text_embedding.flower_dataset import FlowerTextEmbeddingDataset as FlowerDataset 
from models.encoders.text_encoder import HybridCNN
from models.loss.text_embedding_loss import joint_embedding_loss, Fvt
from utils.config import cfg 
from utils.summary import AverageMeter, LogSummary
from utils.timer import Timer



def eval(text_encoder, eval_dataset):
    all_vectors = []
    
    with torch.no_grad():
        for clas_txts, clas, i, lbl in eval_dataset.get_captions_and_info():
            text_vecotrs = text_encoder(clas_txts)
            
            output_dir = os.path.join(cfg.EVAL.TEXT_EMBEDDING.OUTPUT_DIR, clas)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            h5_fn = os.path.join(output_dir, str(i) + '.h5')
            with h5py.File(h5_fn, 'w') as h5fp:
                for j in range(text_vecotrs.shape[0]):
                    h5fp[str(j)] = text_vecotrs[j].cpu().numpy()

                    if lbl < 20:
                        all_vectors.append(text_vecotrs[j])

            if i % cfg.EVAL.TEXT_EMBEDDING.LOG_INTERVAL == 0:
                print("clas {}, index {} done.".format(clas, i))
                sys.stdout.flush()



        all_vectors = torch.stack(all_vectors, dim=0)
        sim_matrix = torch.zeros((all_vectors.shape[0] // 5, all_vectors.shape[0] // 5)).to(cfg.DEVICE)
        print("Begin calculate similarity matrix.")
        for i in range(sim_matrix.shape[0]):
            print("progress: {:.2f}".format(i / sim_matrix.shape[0]))
            for j in range(sim_matrix.shape[1]):
                sim_matrix[i][j] = Fvt(all_vectors[i * 5].reshape(1, -1), all_vectors[j * 5].reshape(1, -1))
    return sim_matrix.cpu().numpy()


# def cal_precise(sim_matrix, instance_num, label_num):
#     labels = np.argsort( - sim_matrix, axis=1)
#     instance_p = []
#     label_p = []

#     for i in labels.shape[0]:



def visual_similarity(dataset):
    
    text_vectors = []
    image_vectors = []
    for clas_txts, img, clas, i, lbl in dataset.get_captions_and_image():
        text_vectors.append(clas_txts)
        image_vectors.append(img)

        if lbl == 20:
            break

    text_vectors = torch.cat(text_vectors, dim=0)
    image_vectors = torch.cat(image_vectors, dim=0)
    print("Begin calculate similarity matrix.")
    sim_matrix = torch.zeros((text_vectors.shape[0] // 5, image_vectors.shape[0] // 5)).to(cfg.DEVICE)
    for i in range(sim_matrix.shape[0]):
        print("progress: {:.2f}".format(i / sim_matrix.shape[0]))
        for j in range(sim_matrix.shape[1]):
            sim_matrix[i][j] = Fvt(text_vectors[i * 5].reshape(1, -1), image_vectors[j * 5].reshape(1, -1))

    return sim_matrix.cpu().numpy()


def main():

    if cfg.EVAL.TEXT_EMBEDDING_DATASET == 'cub':
        eval_dataset = CUBDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.DATASET_DIR,
                              avail_class_fn=cfg.DATASET.TEXT_EMBEDDING_CUB.AVAIL_CLASS_FN,
                              image_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.IMAGE_DIR,
                              text_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_CUTOFF,
                              device=cfg.DEVICE, minibatch_size=cfg.EVAL.TEXT_EMBEDDING.BATCH_SIZE,
                              text_emb_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_EMD_DIR)
    elif cfg.EVAL.TEXT_EMBEDDING_DATASET == 'flower':
        eval_dataset = FlowerDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.DATASET_DIR,
                              avail_class_fn=cfg.DATASET.TEXT_EMBEDDING_FLOWER.AVAIL_CLASS_FN,
                              image_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.IMAGE_DIR,
                              text_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_CUTOFF,
                              device=cfg.DEVICE, minibatch_size=cfg.EVAL.TEXT_EMBEDDING.BATCH_SIZE,
                              text_emb_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_EMD_DIR)
    
    text_encoder = HybridCNN(vocab_dim=eval_dataset.vocab_len, conv_channels=cfg.MODEL.CRNN.CONV_CHANNELS,
                             conv_kernels=cfg.MODEL.CRNN.CONV_KERNELS, conv_strides=cfg.MODEL.CRNN.CONV_STRIDES,
                             rnn_bidir=cfg.MODEL.CRNN.RNN_BIDIR, conv_dropout=cfg.MODEL.CRNN.CONV_DROPOUT,
                             lin_dropout=cfg.MODEL.CRNN.LIN_DROPOUT, rnn_dropout=cfg.MODEL.CRNN.RNN_DROPOUT,
                             rnn_hidden_size=cfg.MODEL.CRNN.RNN_HIDDEN_SIZE // (1 + int(cfg.MODEL.CRNN.RNN_BIDIR)),
                             rnn_num_layers=cfg.MODEL.CRNN.RNN_NUM_LAYERS, lstm=cfg.MODEL.CRNN.LSTM)\
                                .to(cfg.DEVICE).eval()

    if cfg.EVAL.TEXT_EMBEDDING.PRETRAINED_MODEL:
        if os.path.exists(cfg.EVAL.TEXT_EMBEDDING.PRETRAINED_MODEL):
            print("Loading pretrained model from {}".format(cfg.EVAL.TEXT_EMBEDDING.PRETRAINED_MODEL))
            text_encoder.load_state_dict(torch.load(cfg.EVAL.TEXT_EMBEDDING.PRETRAINED_MODEL))
        else:
            print("Pretrained model not exist.")
            exit(1)
    else:
        print("Forget to set the pretrained model path?")
        exit(1)

    # sim_matrix = eval(text_encoder, eval_dataset)

    # h5_fn = os.path.join(cfg.EVAL.TEXT_EMBEDDING.OUTPUT_DIR, 'sim_matrix.h5')
    # with h5py.File(h5_fn, 'w') as h5fp:
    #     h5fp['similarity'] = sim_matrix
    # cmap = sns.color_palette('Reds')
    # fig = sns.heatmap(sim_matrix, cmap=cmap)
    # sns_plot = fig.get_figure()
    # sns_plot.savefig(os.path.join(cfg.EVAL.TEXT_EMBEDDING.OUTPUT_DIR, 'sim_matrix.png'), dpi=400)

    sim_matrix = visual_similarity(eval_dataset)

    h5_fn = os.path.join(cfg.EVAL.TEXT_EMBEDDING.OUTPUT_DIR, 'text_img_sim_matrix.h5')
    with h5py.File(h5_fn, 'w') as h5fp:
        h5fp['similarity'] = sim_matrix
    cmap = sns.color_palette('Reds')
    fig = sns.heatmap(sim_matrix, cmap=cmap)
    sns_plot = fig.get_figure()
    sns_plot.savefig(os.path.join(cfg.EVAL.TEXT_EMBEDDING.OUTPUT_DIR, 'text_img_sim_matrix.png'), dpi=400)


    # precise = cal_precise(sim_matrix)


if __name__ == '__main__':
    main()