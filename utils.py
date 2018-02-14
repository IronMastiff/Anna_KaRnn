import numpy as np

def load_data( dir ):
    with open( dir, 'r' ) as f:
        text = f.read()
    # vocab1 = set( text )    #set( text )去掉字符串中重复的元素， 只保留一个
    vocab = sorted( set( text ) )    #sorted 把数组按从小到大顺序排列
    vocab_to_int = { c : i for i, c in enumerate( vocab ) }    #enumerate把列表转化成索引序列   这里是把他们建立一个表
    int_to_vocab = dict( enumerate( vocab ) )
    encoded = np.array( [vocab_to_int[c] for c in text], dtype = np.int32 )    #把文字转化成数字
    return encoded, int_to_vocab, vocab, vocab_to_int


if __name__ == '__main__':
    data_dir = './data/Ann.txt'
    encoded, int_to_vocab = load_data( data_dir )