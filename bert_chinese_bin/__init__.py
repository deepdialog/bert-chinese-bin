import os
from tf_text_model import BertTokenizer, BertEmbedding

CURRENT_DIR = os.path.realpath(
    os.path.dirname(__file__)
    if '__file__' in dir() else
    os.path.dirname('.'))
BERT_DIR = os.path.join(
    CURRENT_DIR, 'chinese_roberta_wwm_ext_L-12_H-768_A-12/')


class BertChineseTokenizer(BertTokenizer):
    def __init__(self):
        super(BertChineseTokenizer, self).__init__(
            os.path.join(BERT_DIR, 'vocab.txt'))


class BertChineseEmbedding(BertEmbedding):
    def __init__(self, trainable=False):
        super(BertChineseEmbedding, self).__init__(
            bert_params=BERT_DIR, trainable=trainable, model_dir=BERT_DIR)
