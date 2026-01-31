import sys
import os
import argparse
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
try:
    from . import Load as iload
    from . import Read as iread
    from . import Blast as iblast
    from . import Visual as ivis
    from . import SVM as isvm
    from . import Version
except:
    import Load as iload
    import Read as iread
    import Blast as iblast
    import Visual as ivis
    import SVM as isvm
    import Version

iload.load_blast()


# fuctions ####################################################################
def parse_version(args):
    print('\nIRAP version=' + Version.version)


# make database
def parse_makedb(args):
    print('\n>>>Making database...\n')
    iblast.blast_makedb(args.file[0], args.output[0])


# check database
def parse_checkdb(args):
    print('\n>>>Checking database...\n')
    iblast.blast_chackdb(args.file[0])


# read
def parse_read(args):
    print('\n>>>Reading files...\n')
    if len(args.file) == len(args.output):
        for eachdir in range(len(args.file)):
            iread.read_read(args.file[eachdir], args.output[eachdir])
    else:
        print('\n>>>ERROR: The number of input files and output folders is not equal.\n')


# psi-blast
def parse_blast(args):
    print('\n>>>Blasting PSSM matrix...\n')
    if len(args.folder) == len(args.output):
        for eachdir in range(len(args.folder)):
            iread.read_blast(args.folder[eachdir], args.database[0], int(args.num_iterations[0]), float(args.expected_value[0]), args.output[eachdir])
    else:
        print('\n>>>ERROR: The number of input folders and output folders is not equal.\n')


# extract features
def parse_extract(args):
    print('\n>>>Extracting PSSM matrix features...\n')
    if args.reduce_aa:
        iread.read_extract_raabook(args.folder[0], args.folder[1], args.output[0], args.reduce_aa[0])
    else:
        iread.read_extract_selfraa(args.folder[0], args.folder[1], args.output[0], args.self_raac[0])

# search best factors
def parse_grid(args):
    if args.document:
        iread.read_grid_folder([args.document[0]], model_type=args.model)
    else:
        iread.read_grid_folder(args.folder[0], model_type=args.model)

# argparse ####################################################################
def irap():
    parser = argparse.ArgumentParser(description='An Intelligent RAAC-PSSM Protein Prediction Package',
                                     fromfile_prefix_chars='@', conflict_handler='resolve')
    subparsers = parser.add_subparsers(help='IRAP help')
    # make database
    parser_ma = subparsers.add_parser('makedb', add_help=False, help='make database')
    parser_ma.add_argument('file', nargs=1, help='fasta database name')
    parser_ma.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_ma.set_defaults(func=parse_makedb)
    # check database
    parser_cd = subparsers.add_parser('checkdb', add_help=False, help='check database and remove repetitive sequences')
    parser_cd.add_argument('file', nargs=1, help='fasta database name')
    parser_cd.set_defaults(func=parse_checkdb)
    # read and segment original files
    parser_re = subparsers.add_parser('read', add_help=False, help='read protein sequences files and segment it')
    parser_re.add_argument('file', nargs='+', help='fasta file paths')
    parser_re.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_re.set_defaults(func=parse_read)
    # blast PSSM matrix
    parser_bl = subparsers.add_parser('blast', add_help=False, help='get PSSM matrix by psi-blast')
    parser_bl.add_argument('folder', nargs='+', help='input a folder containing single sequence files')
    parser_bl.add_argument('-db', '--database', nargs=1, type=str, help='database for blast')
    parser_bl.add_argument('-n', '--num_iterations', nargs=1, type=str, help='number of blast cycles')
    parser_bl.add_argument('-ev', '--expected_value', nargs=1, type=str, help='expected value of blast cycles')
    parser_bl.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_bl.set_defaults(func=parse_blast)
    # extract features
    parser_ex = subparsers.add_parser('extract', add_help=False, help='extract the features of PSSM matrix')
    parser_ex.add_argument('folder', nargs=2, help='input PSSM matrix files folder')
    parser_ex.add_argument('-raa', '--reduce_aa', nargs=1, type=str, help='reduce amino acid file')
    parser_ex.add_argument('-o', '--output', nargs=1, type=str, help='output folder')
    parser_ex.add_argument('-l', '--lmda', nargs=1, type=str, help='sliding window lmda')
    parser_ex.add_argument('-r', '--self_raac', nargs=1, type=str, help='self raac')
    parser_ex.set_defaults(func=parse_extract)
    # grid search
    parser_se = subparsers.add_parser('search', add_help=False, help='search c_number and gamma for training')
    parser_se.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_se.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_se.add_argument('-m', '--model', help='rfci or rmic')
    parser_se.set_defaults(func=parse_grid)
    
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        pass


# main
if __name__ == '__main__':
    irap()
