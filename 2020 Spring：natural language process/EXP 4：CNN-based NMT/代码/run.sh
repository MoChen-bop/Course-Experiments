#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --exp-name=$2 --cuda
elif [ "$1" = "train_with_contex_cnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --exp-name=$2 --with-contex --cuda
elif [ "$1" = "train_with_contex_lstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --exp-name=$2 --with-contex --contex-LSTM --cuda
elif [ "$1" = "train_with_contex_lstm_multi_encoder" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --exp-name=$2 --with-contex --contex-LSTM --multi-encoder --cuda
elif [ "$1" = "train_multi_encoder" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --exp-name=$2 --multi-encoder --cuda


elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/$2/test_outputs.txt --exp-name=$2 --cuda
elif [ "$1" = "test_with_contex_cnn" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/$2/test_outputs.txt --exp-name=$2 --with-contex --cuda
elif [ "$1" = "test_with_contex_lstm" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/$2/test_outputs.txt --exp-name=$2 --with-contex --contex-LSTM --cuda
elif [ "$1" = "test_with_contex_lstm_multi_encoder" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/$2/test_outputs.txt --exp-name=$2 --with-contex --contex-LSTM --multi-encoder --cuda
elif [ "$1" = "test_with_contex_lstm_multi_encoder_dev" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/dev.es ./en_es_data/dev.en outputs/$2/dev/test_outputs.txt --exp-name=$2 --with-contex --contex-LSTM --multi-encoder --cuda
elif [ "$1" = "test_multi_encoder" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/$2/test_outputs.txt --exp-name=$2 --multi-encoder --cuda


elif [ "$1" = "train_local_q1" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder --cuda
elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q1.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt \
        --no-char-decoder --cuda

elif [ "$1" = "train_local_q2" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100 --cuda
elif [ "$1" = "train_local_q2_with_contex_cnn" ]; then
    python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100 --exp-name=$2 --with-contex --cuda
elif [ "$1" = "train_local_q2_with_contex_lstm" ]; then
    python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100 --exp-name=$2 --with-contex --contex-LSTM --cuda
elif [ "$1" = "train_local_q2_with_contex_lstm_multi_encoder" ]; then
    python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100 --exp-name=$2 --with-contex --contex-LSTM --multi-encoder --cuda

elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/$2/test_outputs_local_q2.txt --cuda
elif [ "$1" = "test_local_q2_with_contex_cnn" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/$2/test_outputs_local_q2.txt --exp-name=$2 --with-contex --cuda
elif [ "$1" = "test_local_q2_with_contex_lstm" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/$2/test_outputs_local_q2.txt --exp-name=$2 --with-contex --contex-LSTM --cuda
elif [ "$1" = "test_local_q2_with_contex_lstm_multi_encoder" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/$2/test_outputs_local_q2.txt --exp-name=$2 --with-contex --contex-LSTM --multi-encoder --cuda



elif [ "$1" = "vocab" ]; then
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --size=200 --freq-cutoff=1 vocab_tiny_q1.json
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        vocab_tiny_q2.json
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
