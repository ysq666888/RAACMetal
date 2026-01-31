from typing import Union

ion_reduce_types: dict = {
    'fe2': 'type_51+size_19+LI-V-G-A-P-Q-N-M-T-S-C-E-D-K-R-Y-F-W-H',
    'co': 'type_63+size_8+QSNTGAP-ED-LIVM-FWY-H-R-K-C', 
    'mg': 'type_11+size_10+VILMF-WY-A-C-ED-RK-G-P-ST-HQN', 
    'ni': 'type_51+size_19+LI-V-G-A-P-Q-N-M-T-S-C-E-D-K-R-Y-F-W-H',
    'fe': 'type_7+size_11+W-C-G-P-H-NDE-RQK-AST-F-Y-VMIL', 
    'cu': 'type_32+size_15+RK-QE-N-D-H-S-T-P-A-G-IV-L-M-FYW-C', 
    'mn': 'type_1+size_15+LVIM-C-A-G-S-T-P-FY-W-E-D-N-Q-KR-H', 
    'k': 'type_49+size_17+C-FY-W-ML-IV-G-P-A-T-S-N-H-Q-E-D-R-K', 
    'zn': 'type_63+size_8+QSNTGAP-ED-LIVM-FWY-H-R-K-C',
    'na': 'type_38+size_9+FWYH-ML-IV-CA-NTS-P-G-DE-QRK' ,
    'ca': 'type_33+size_11+ST-A-ND-G-RQ-EK-H-P-IVLM-WYF-C' 
}

class IrapSeq:
    def __init__(self, irap_path:str) -> None:
        self.irap_path: str = irap_path
        self.iraps: dict = self.__readirap__()
    
    def __readirap__(self) -> dict[str, str]:
        irap: dict[str, str] = dict()
        with open(self.irap_path, 'r') as file:
            for line in file:
                line = line.strip().split(' ')
                the_type: str = line[1]
                the_size: str = line[3]
                context: str = line[-1]
                name = f'type_{the_type}+size_{the_size}+{context}'
                assert name not in irap.keys(), f'{name} 重复'
                irap[name] = context
        return irap
        
    def irap_dict(self, type:str, size: str) -> str:
        name: str = f'type:{type}+size:{size}'
        return self.iraps[name]
    
    def irap_dicts(self) -> dict[str, str]:
        return self.iraps
        
    def irap(self, seq:str, type_and_size: Union[bool, str] = None) -> str:
        if type_and_size:
            name: str = type_and_size
        else:
            name: str = f'type:0+size:1'
        irap_context: list = self.iraps[name].split("-")
        return self.__seqtoirap__(seq.upper(), irap_context)
        
    @staticmethod
    def __seqtoirap__(seq: str, irap_list: list) -> str:
        irap_seq: str = ''
        for res in seq:
            for irap_type in irap_list:
                if res in irap_type:
                    irap_seq += irap_type[0]
        return irap_seq



class FullTokenizer(object):

	def __init__(self, vocab_file: str) -> None:
		self.vocab: dict[str, int] = dict()
		with open(vocab_file, 'r') as vocab_file:
			vocab_lines: list[str] = vocab_file.readlines()
			index: int = 0
			for vocab_line in vocab_lines:
				token: str = vocab_line.strip()
				self.vocab[token] = index
				index += 1

	def get_vocab(self) -> dict[str, int]:
		return self.vocab


def select_model(target_name:str):
    reduce_type = ion_reduce_types[target_name]
    return f"./Models_{target_name}/23_model_{reduce_type}"