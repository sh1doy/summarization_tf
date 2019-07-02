# Attention-based Tree-to-Sequence Code Summarization Model

The TensorFlow Eager Execution implementation of [Source Code Summarization with Extended Tree-LSTM](https://arxiv.org/abs/1906.08094) (Shido+, 2019)

including:

- **Multi-way Tree-LSTM model (Ours)**
- Child-sum Tree-LSTM model
- N-ary Tree-LSTM model
- DeepCom (Hu et al.)
- CODE-NN (Iyer et al.)

## Dataset

1. Download raw dataset from [https://github.com/xing-hu/DeepCom]
2. Parse them with parser.jar

## Usage

1. Prepare tree-structured data with `dataset.py`
    - Run `$ python dataset.py [dir]`
2. Train and evaluate model with `train.py`
    - See `$ python train.py -h`
