# import package
import os
now_path = os.getcwd()
import numpy as np
import urllib.request
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import SVM as isvm
import tarfile
import gzip
import platform


# load fasta file
def load_fasta_file(file):
    with open(file, 'r') as f:
        data = f.readlines()[-1]
    return data.strip()


# load svm file
def load_svmfile(file):
    with open(file, 'r') as f1:
        data = f1.readlines()
    # 提取特征list
    features = []
    features_label = []
    for i in data:
        line = i.strip('\n').split(' ')
        fs_box = line[1:]
        mid_box = []
        for j in fs_box:
            mid_box.append(float(j.split(':')[-1]))
        features.append(mid_box)
        features_label.append(int(line[0]))
    # 转换为数组
    np_data = np.array(features)
    np_label = np.array(features_label)
    return np_data, np_label


# load svm path from numpy
def load_np_path(np_data, np_label, out=now_path):
    file = ''
    for i in range(len(np_label)):
        mid = str(np_label[i])
        for j in range(len(np_data[i])):
            mid += ' ' + str(j+1) + ':' + str(np_data[i,j])
        file += mid + '\n'
    if out != now_path:
        with open(out, 'w') as f:
            f.write(file[:-1])
        return out
    else:
        with open(os.path.join(now_path, 'save_file.txt'), 'w') as f:
            f.write(file[:-1])
        return os.path.join(now_path, 'save_file.txt')


# load sequence data set
def load_fasta(file, out=None):
    with open(file, 'r') as u:
        lines = u.readlines()
    result = ''
    for i in lines:
        i = i.strip('\n')
        if i and i[0] == '>':
            result = result + '\n' + i + '\n'
        else:
            result = result + i
    result = result[1:].split('\n')
    sq_dic = {}
    for i in range(len(result)-1):
        if '>' in result[i]:
            sq_dic[result[i]] = result[i+1]
    if out == None:
        return sq_dic
    else:
        t = 0
        path = []
        for i in range(len(result)-1):
            if '>' in result[i]:
                t += 1
                line = result[i] + '\n' + result[i+1]
                path.append(os.path.join(out, str(t) + '.fasta'))
                with open(os.path.join(out, str(t) + '.fasta'), 'w') as f:
                    f.write(line)
        return path


# load sequence data set to default folder
def load_fasta_folder(file=None, out='positive'):
    if file != None:    
        sq_path = os.path.join(now_path, 'Reads')
        if 'Reads' not in os.listdir(now_path):
            os.makedirs(sq_path)
        next_path = os.path.join(sq_path, out)
        if out not in os.listdir(sq_path):
            os.makedirs(next_path)
        sq = load_fasta(file, out=next_path)
        return sq
    else:
        sq = []
        for i in os.listdir(out):
            sq.append(os.path.join(out, i))
        return sq


# reload pssm folder
def load_reload_folder(path):
    out = []
    for i in os.listdir(path):
        out.append(os.path.join(path, i))
    return out


# load raac dictionary
def load_raac(file):
    with open(file, 'r') as code:
        raacode = code.readlines()
    raa_dict = {}
    raa_index = []
    for eachline in raacode:
        each_com = eachline.strip('\n').split()
        raa_com = each_com[-1].split('-')
        raa_ts = 't' + each_com[1] + 's' + each_com[3]
        raa_id = str(raacode.index(eachline))
        if len(raa_id) == 1:
            raa_id = '000' + raa_id
        if len(raa_id) == 2:
            raa_id = '00' + raa_id
        if len(raa_id) == 3:
            raa_id = '0' + raa_id
        raa_ts = raa_id + '-' + raa_ts
        raa_dict[raa_ts] = raa_com
        raa_index.append(raa_ts)
    return raa_dict, raa_index


# read pssm matrix
def load_pssm(path):
    with open(path, 'r') as f:
        data = f.readlines()
    matrix = []
    aa_id = []
    end_matrix = 0
    for j in data:
        if 'Lambda' in j and 'K' in j:
            end_matrix = data.index(j)
            break
    for eachline in data[3:end_matrix - 1]:
        row = eachline.split()
        newrow = row[0:22]
        for k in range(2, len(newrow)):
            newrow[k] = int(newrow[k])
        nextrow = newrow[2:]
        matrix.append(nextrow)
        aa_id.append(newrow[1])
    return matrix, aa_id


# load selected feature of svm file
def load_svm_feature(file, filter_index, number, out=None):
    # 读取特征排序以及特征文件
    with open(filter_index, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    index = data[0].split(' ')[1:-1]
    with open(file, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    # 提取矩阵特征
    type_f = []
    matrix = []
    for line in data:
        line = line.split(' ')
        type_f.append(line[0])
        mid_box = line[1:]
        for i in range(len(mid_box)):
            mid_box[i] = mid_box[i].split(':')[-1]
        matrix.append(mid_box)
    # 提取数组
    out_type = np.zeros(len(type_f))
    out_matrix = np.zeros([len(matrix), number])
    for i in range(len(type_f)):
        out_type[i] = int(type_f[i])
        for j in range(number):
            out_matrix[i][j] = float(matrix[i][int(index[j])-1])
    if out == None:
        return out_matrix, out_type
    else:
        content = ''
        for i in range(len(out_matrix)):
            line = out_matrix[i]
            mid = str(int(out_type[i]))
            for j in range(number):
                mid += ' ' + str(j+1) + ':' + str(line[j])
            content += mid + '\n'
        with open(out, 'w') as f:
            f.write(content[:-1])
        return out_matrix, out_type


# load selected feature of numpy
def load_numpy_feature(matrix, type_f, number, index=None, out=None):
    if index == None:
        out_type = np.zeros(len(type_f))
        out_matrix = np.zeros([len(matrix), number])
        for i in range(len(type_f)):
            out_type[i] = type_f[i]
            for j in range(number):
                out_matrix[i][j] = matrix[i][j]
    else:
        out_type = np.zeros(len(type_f))
        out_matrix = np.zeros([len(matrix), number])
        for i in range(len(type_f)):
            out_type[i] = type_f[i]
            for j in range(number):
                out_matrix[i][j] = matrix[i][index[j]]
    if out == None:
        return out_matrix, out_type
    else:
        content = ''
        for i in range(len(out_matrix)):
            line = out_matrix[i]
            mid = str(out_type[i])
            for j in range(number):
                mid += ' ' + str(j+1) + ':' + str(line[j])
            content += mid + '\n'
        with open(out, 'w') as f:
            f.write(content[:-1])
        return out_matrix, out_type


# read hyperparameters file
def load_hys(file):
    with open(file, 'r') as f:
        data = f.readlines()
    cg_box = {}
    for i in data:
        cg_box[i.strip('\n').split('\t')[0]] = [float(i.strip('\n').split('\t')[1].split(': ')[-1]), float(i.strip('\n').split('\t')[2].split(': ')[-1])]
    return cg_box


# read raac of different types
def load_ssc(raa_file, type_r):
    with open(raa_file, "r") as f:
        data = f.readlines()
    out_box = []
    for line in data:
        line = line.strip("\n").split(" ")
        if line[1] == type_r:
            out_box.append(line[4])
    all_sq = ""
    for i in out_box[0]:
        if i != "-":
            all_sq += i + "-"
    out_box.append(all_sq[:-1])
    for i in range(len(out_box)):
        out_box[i] = out_box[i].split("-")
    return out_box[::-1]


# read pssm weblogo
def load_weblogo(matrix):
    def line_score(newrow):
        out_box = []
        a = 0
        for i in newrow:
            if i > 0:
                a += i
        for i in newrow:
            if i > 0:
                out_box.append(i * 100 / a)
            else:
                out_box.append(0)
        return out_box
    # main
    out = []
    for eachline in matrix:
        newrow = line_score(eachline)
        for i in range(len(newrow)):
            newrow[i] = int(newrow[i])
        out.append(newrow)
    return out


def load_un_gz(file_name):
    f_name = file_name.strip(".gz")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


# load blast database
def load_pdbaa():
    if 'pdbaa.pdb' not in os.listdir(os.path.join(file_path, 'blastDB')):
        url = 'https://ftp.ncbi.nlm.nih.gov/blast/db/pdbaa.tar.gz'
        save_path = os.path.join(file_path, 'pdbaa.tar.gz')
        if 'pdbaa.tar.gz' not in os.listdir(file_path):
            urllib.request.urlretrieve(url, filename=save_path)
        load_un_gz(save_path)
        t = tarfile.open(save_path.strip(".gz"))
        t.extractall(path=os.path.join(file_path, 'blastDB'))
        t.close()
        os.remove(save_path.strip(".gz"))
        print('\npdbaa database has been loaded successfully!')
    else:
        print('\npdbaa database has been loaded successfully!')


# save the feature file to model file
def load_model_save_file(file, c=8, g=0.125, out=now_path):
    if out != now_path:
        isvm.svm_train(file, c, g, out)
        return out
    else:
        isvm.svm_train(file, c, g, os.path.join(out, 'models.model'))
        return os.path.join(out, 'models.model')


# save the feature folder to model folder
def load_model_save_folder(path, cg=None, out=now_path):
    model_path = []
    if out != now_path:
        if os.path.split(out)[-1] not in os.listdir(os.path.split(out)[0]):
            os.makedirs(out)
        if cg != None:
            cg_box = load_hys(cg)
            for i in path:
                if cg_box[i][1] == 0:
                    cg_box[i][1] = 0.01
                isvm.svm_train(i, cg_box[i][0], cg_box[i][1], os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
                model_path.append(os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
        else:
            for i in path:
                isvm.svm_train(i, 8, 0.125, os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
                model_path.append(os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
    else:
        if cg != None:
            cg_box = load_hys(cg)
            for i in path:
                if cg_box[i][1] == 0:
                    cg_box[i][1] = 0.01
                isvm.svm_train(i, cg_box[i][0], cg_box[i][1], os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
                model_path.append(os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
        else:
            for i in path:
                isvm.svm_train(i, 8, 0.125, os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
                model_path.append(os.path.join(out, os.path.split(i)[-1].split('.')[0] + '.model'))
    return model_path


# load precaution
def load_precaution():
    with open(os.path.join(file_path, 'README'), 'r', encoding='UTF-8') as f:
        prec_data = f.read()
    return prec_data


# load blast database
def load_blast():
    if platform.system() == 'Windows':
        file1 = 'psiblast.exe'
        file2 = 'makeblastdb.exe'
        file3 = 'nghttp2.dll'
        if file3 not in os.listdir(os.path.join(file_path, 'bin')):
            url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file3
            save_path = os.path.join(os.path.join(file_path, 'bin'), file3)
            urllib.request.urlretrieve(url, filename=save_path)
            print('\nconfiguration file has been loaded!')
    else:
        file1 = 'psiblast'
        file2 = 'makeblastdb'
    file4 = 'pdbaa.tar.gz'
    file5 = 'README'
    if file1 not in os.listdir(os.path.join(file_path, 'bin')):
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file1
        save_path = os.path.join(os.path.join(file_path, 'bin'), file1)
        urllib.request.urlretrieve(url, filename=save_path)
        print('\npsiblast function has been loaded!')
    if file2 not in os.listdir(os.path.join(file_path, 'bin')):
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file2
        save_path = os.path.join(os.path.join(file_path, 'bin'), file2)
        urllib.request.urlretrieve(url, filename=save_path)
        print('\nmakeblastdb function has been loaded!')
    if file4 not in os.listdir(file_path):
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file4
        save_path = os.path.join(file_path, file4)
        urllib.request.urlretrieve(url, filename=save_path)
        load_un_gz(save_path)
        t = tarfile.open(save_path.strip(".gz"))
        t.extractall(path=os.path.join(file_path, 'blastDB'))
        t.close()
        os.remove(save_path.strip(".gz"))
        print('\npdbaa database has been loaded!')
    if file5 not in os.listdir(file_path):
        url = 'http://bioinfor.imu.edu.cn/rpct/static/data/' + file5
        save_path = os.path.join(file_path, file5)
        urllib.request.urlretrieve(url, filename=save_path)
        print('\nprecaution has been loaded!')
    