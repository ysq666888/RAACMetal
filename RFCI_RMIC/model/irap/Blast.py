# import package
import os
import subprocess
now_path = os.getcwd()
blast_path = os.path.dirname(__file__)
database_path = os.path.join(blast_path, 'blastDB')
function_path = os.path.join(blast_path, 'bin')
import sys
sys.path.append(blast_path)
from Visual import visual_longcommand, visual_easy_time

# original psiblast
def blast_psiblast(file, database, number, ev, out=now_path, vi=False):
    save_path = os.path.join(out, os.path.split(file)[-1].split('.')[0] + '.pssm')
    database_path = os.path.join(os.path.join(blast_path, 'blastDB'), database)
    command = visual_longcommand(file, database_path, number, ev, save_path)
    if vi == True:
        print(command)
    outcode = subprocess.Popen(command, shell=True)
    if outcode.wait() != 0:
        print('\r\tProblems', end='', flush=True)
    if 'A' in os.listdir(out):
        os.remove(os.path.join(out, 'A'))
    if 'A' in os.listdir(now_path):
        os.remove(os.path.join(now_path, 'A'))
    return save_path

# original psiblast
def blast_psiblast_folder(folder, database, number, ev, name='positive', vi=False):
    blast_path = os.path.join(now_path, 'PSSMs')
    if 'PSSMs' not in os.listdir(now_path):
        os.makedirs(blast_path)
    next_path = os.path.join(blast_path, name)
    if name not in os.listdir(blast_path):
        os.makedirs(next_path)
    pssm_path = []
    start_e = 0
    for i in folder:
        start_e += 1
        visual_easy_time(start_e, len(folder))
        pssm_path.append(blast_psiblast(i, database, number, ev, out=next_path, vi=vi))
    return pssm_path

# database standardization
def blast_makedb(file, name, out=database_path):
    save_path = os.path.join(out, name)
    command = os.path.join(function_path, 'makeblastdb') + ' -in ' + file + ' -dbtype prot -parse_seqids -out ' + save_path
    outcode = subprocess.Popen(command, shell=True)
    outcode.wait()

# database deduplication
def blast_chackdb(file, out=now_path):
    f = open(file, 'r', encoding='UTF-8')
    namebox = []
    eachsequence = ''
    line = 1
    t = 0
    while line:
        t += 1
        line = f.readline()
        if '>' in line:
            if line.split(' ')[0] not in namebox and line.split(' ')[0].upper() not in namebox:
                namebox.append(line.split(' ')[0].upper())
                eachsequence += line.split(' ')[0].upper() + '\n'
                writeable = 'Ture'
            else:
                writeable = 'False'
        else:
            if writeable == 'Ture':
                eachsequence += line
            else:
                pass
        if t == 20000:
            with open(os.path.join(out, 'ND_' + file), 'a', encoding='UTF-8') as o:
                o.write(eachsequence)
                o.close()
            eachsequence = ''
            t = 0
    with open(os.path.join(out, 'ND_' + file), 'a', encoding='UTF-8') as o:
        o.write(eachsequence)
        o.close()
    f.close()
    return os.path.join(out, 'ND_' + file)
