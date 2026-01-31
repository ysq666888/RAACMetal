import tensorflow as tf
import os
import argparse
from data_make.utils import *


def read_fasta(fasta_file:str) -> dict:
    output:dict = dict()
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                protein_name = line.strip()
                seq = next(f).strip()
                output[protein_name] = seq
    return output


def sliding_window(seqs:dict, window_size=25, pad_char='-') -> list:
    half_window = window_size // 2
    protein_data:list = []
    for protein_name, seq in seqs.items():
        reduce_seq = seq['reduce_seq']
        origin_seq = seq['origin_seq']
        padded_sequence = pad_char * half_window + reduce_seq + pad_char * half_window
        for i in range(len(reduce_seq)):
            fragment_data = {
                'protein_name':protein_name,
                'center_res': origin_seq[i],
                'fragment': [x.replace('-', 'PAD') for x in padded_sequence[i:i+window_size]]
            }
            protein_data.append(fragment_data)
    return protein_data


def convert_tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:

	ids: list = []
	for position in range(len(tokens)):
		if tokens[position] in vocab.keys():
			ids += [vocab[tokens[position].upper()]]
		else:
			ids += [vocab['-']]
	return ids


def get_features(
		seq: list[str],
		vocab: dict[str, int],
		max_length: int=27):

	max_length: int = max_length - 2

	pad_id: int = vocab["[PAD]"]
	start: int = vocab["[CLS]"]
	end: int = vocab["[SEP]"]


	seq_id: list[int] = convert_tokens_to_ids(seq, vocab)

	seq_len: int = len(seq_id)
	assert max_length - seq_len >= 0
    
	seq_id: list[int] = [start] + seq_id + [end]

	pad: list[int] = [0 if x == pad_id else 1 for x in seq_id]

	return seq_id, pad


def tokenize(
		protein_data: list,
		vocab_file: str) -> None:

	vocabs: dict[str, int] = FullTokenizer(vocab_file).get_vocab()
	with tf.io.TFRecordWriter(f'./temp.tfrecord') as writer:
		for single_protein in protein_data:
			protein_name = single_protein['protein_name'].encode('utf-8')
			center_res = single_protein['center_res'].encode('utf-8')
			seq_id, pad = get_features(
				single_protein['fragment'],
				vocabs)
			features: dict[str, tf.Tensor] = {
				'protein_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[protein_name])),
                'center_res': tf.train.Feature(bytes_list=tf.train.BytesList(value=[center_res])),
				'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=seq_id)),
				'input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=pad)),
			}
			example = tf.train.Example(features=tf.train.Features(feature=features))
			writer.write(example.SerializeToString())
		writer.close()


def reduce_fasta(seqs: str, reduce_type):
    irap_path = os.path.join('.', 'data_make', 'irap.txt')
    irap = IrapSeq(irap_path=irap_path)
    for key, value in seqs.items():
        seqs[key] = {'reduce_seq':irap.irap(seq=value, type_and_size=reduce_type), 'origin_seq':value}
    return seqs


def main(fasta_file: str, target:str):
    reduce_type = ion_reduce_types[target]
    seqs:dict = read_fasta(fasta_file=fasta_file)
    irap_seqs:dict = reduce_fasta(seqs=seqs, reduce_type=reduce_type)
    protein_data:list = sliding_window(irap_seqs)
    tokenize(protein_data=protein_data, vocab_file='./data_make/vocab.txt')


if __name__ == '__main__':
    allow_ions = ['ca', 'co', 'cu', 'fe', 'fe2', 'k', 'mg', 'mn', 'na', 'ni', 'zn']
    parser = argparse.ArgumentParser(description='Making tfrecord file')
    parser.add_argument('-f', '--fasta_file', help='fasta_file path', required=True)
    parser.add_argument('-t', '--target_ion', choices=allow_ions, help=f'ion ligand predicted\nallow_ion={allow_ions}', required=True)
    args = parser.parse_args()
    fasta_file = args.fasta_file
    target = args.target_ion
    main(fasta_file=fasta_file, target=target)
