# import package
import os
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import Load as iload
import Visual as ivis
import Feature as ifeat
raac_path = os.path.join(file_path, 'raacDB')
now_path = os.getcwd()
extract_method = ['RAAC-PSSM', 'RAAC-KPSSM', 'RAAC-DTPSSM', 'RAAC-SW', 'RAAC-KMER', 'SAAC', 'OAAC']

# extract pssm
def extract_single(path_list, tp):
    pssm_matrix, pssm_aaid, pssm_type = [], [], []
    for file in path_list:
        matrix, aaid = iload.load_pssm(file)
        pssm_matrix.append(matrix)
        pssm_aaid.append(aaid)
        pssm_type.append(tp)
    return pssm_matrix, pssm_aaid, pssm_type

# extract save
def extract_save(max_matrix, max_type, out, raa_type, method_id):
    file = ''
    for i in range(len(max_type)):
        mid = str(max_type[i])
        t = 0
        for j in range(len(max_matrix[i])):
            t += 1
            mid += ' ' + str(t) + ':' + str(max_matrix[i][j])
        file += mid + '\n'
    file_name = os.path.join(out, raa_type + method_id + '.rap')
    with open(file_name, 'w') as f:
        f.write(file)
    return file_name

# extract reduce row
def extract_reduce_row(mid_matrix, mid_aaid, reduce, raa_type):
    out_matrix = []
    raa_list = iload.load_raac(os.path.join(raac_path, reduce))[0][raa_type]
    for i in range(len(mid_aaid)):
        out = ivis.visual_create_nn_matrix(x=len(raa_list), y=len(mid_matrix[i][0]))
        each_matrix = mid_matrix[i]
        each_aaid = mid_aaid[i]
        for j in range(len(each_matrix)):
            for k in range(len(each_matrix[j])):
                for l in range(len(raa_list)):
                    if each_aaid[j] in raa_list[l]:
                        out[l][k] += each_matrix[j][k]
        out_matrix.append(out)
    return out_matrix

# extract reduce row single file
def extract_reduce_row_sf(mid_matrix, mid_aaid, reduce, raa_type):
    raa_list = iload.load_raac(os.path.join(raac_path, reduce))[0][raa_type]
    out = ivis.visual_create_nn_matrix(x=len(raa_list), y=len(mid_matrix[0]))
    for j in range(len(mid_matrix)):
        for k in range(len(mid_matrix[j])):
            for l in range(len(raa_list)):
                if mid_aaid[j] in raa_list[l]:
                    out[l][k] += mid_matrix[j][k]
    return out

# extract reduce col
def extract_reduce_col(mid_matrix, reduce, raa_type):
    out_matrix = []
    raa_list = iload.load_raac(os.path.join(raac_path, reduce))[0][raa_type]
    aa_index = ivis.visual_create_aa()
    for i in range(len(mid_matrix)):
        out = ivis.visual_create_nn_matrix(x=len(mid_matrix[i]), y=len(raa_list))
        each_matrix = mid_matrix[i]
        for j in range(len(aa_index)):
            for k in range(len(each_matrix)):
                for l in range(len(raa_list)):
                    if aa_index[j] in raa_list[l]:
                        out[k][l] += each_matrix[k][j]
        out_matrix.append(out)
    return out_matrix

# extract reduce col single file
def extract_reduce_col_sf(mid_matrix, reduce, raa_type):
    raa_list = iload.load_raac(os.path.join(raac_path, reduce))[0][raa_type]
    aa_index = ivis.visual_create_aa()
    out = ivis.visual_create_nn_matrix(x=len(mid_matrix), y=len(raa_list))
    for j in range(len(aa_index)):
        for k in range(len(mid_matrix)):
            for l in range(len(raa_list)):
                if aa_index[j] in raa_list[l]:
                    out[k][l] += mid_matrix[k][j]
    return out

# extract scale data
def extract_scale(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            min_x = min(matrix[i][j])
            max_x = max(matrix[i][j])
            for k in range(len(matrix[i][j])):
                if (matrix[i][j][k] - min_x) != 0:
                    matrix[i][j][k] = round((matrix[i][j][k] - min_x) / (max_x - min_x), 4)
                else:
                    matrix[i][j][k] = 0
    return matrix

# extract scale data multy file
def extract_scale_mf(matrix):
    for r in range(len(matrix)):
        for i in range(len(matrix[r])):
            for j in range(len(matrix[r][i])):
                min_x = min(matrix[r][i][j])
                max_x = max(matrix[r][i][j])
                for k in range(len(matrix[r][i][j])):
                    if (matrix[r][i][j][k] - min_x) != 0:
                        matrix[r][i][j][k] = round((matrix[r][i][j][k] - min_x) / (max_x - min_x), 4)
                    else:
                        matrix[r][i][j][k] = 0
    return matrix

# extract reduce matrix
def extract_reduce(max_matrix, max_aaid, reduce, raacode):
    out_feature = []
    start_e = 0
    for i in range(len(max_matrix)):
        start_e += 1
        start_n = 0
        each_matrix = max_matrix[i]
        each_aaid = max_aaid[i]
        mid_box = []
        for raa in raacode[1]:
            start_n += 1
            ivis.visual_detal_time(start_e, len(max_matrix), start_n, len(raacode[1]))
            file = extract_reduce_row_sf(each_matrix, each_aaid, reduce, raa)
            file = extract_reduce_col_sf(file, reduce, raa)
            new_box = []
            for j in file:
                new_box += j
            mid_box.append(new_box)
        out_feature.append(mid_box)
    return out_feature

# extract mixed feature and save file
def extract_combine(max_tuble, max_type, out, reduce, raacode, method_id):
    pssm_path = []
    start_e = 0
    for m in range(len(max_tuble)):
        if max_tuble[m] != []:
            break
    for i in range(len(raacode[1])):
        start_e += 1
        ivis.visual_easy_time(start_e, len(raacode[1]))
        raa = raacode[1][i]
        # combine
        mid_box = ivis.visual_create_n_matrix(x=len(max_tuble[m]), fill=[])
        for j in range(len(max_tuble[m])):
            for k in range(len(max_tuble)):
                if max_tuble[k] != []:
                    mid_box[j] = mid_box[j] + max_tuble[k][j][i]
        # save
        path = extract_save(mid_box, max_type, out, raa, method_id)
        pssm_path.append(path)
    return pssm_path

# extract feature to one line
def extract_one(raac_pssm_fs):
    out = []
    for i in raac_pssm_fs:
        mid = []
        for j in i:
            mid += j
        out.append(mid)
    return out

# extract main
def extract_pssm(method=[0,1,2,3,4,5,6], pos=None, neg=None, reduce=False, raaBook=None, out=now_path):
    pssm_path = []
    if os.path.split(out)[-1] not in os.listdir(os.path.split(out)[0]):
        os.makedirs(out)
    if pos != None and neg != None:
        pos_matrix, pos_aaid, pos_type = extract_single(pos, 0)
        neg_matrix, neg_aaid, neg_type = extract_single(neg, 1)
        if reduce != False and raaBook != None:
            raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs = [], [], [], [], [], [], []
            raacode = iload.load_raac(os.path.join(raac_path, raaBook))
            method_id = '-'
            # RAAC-PSSM
            if 0 in method:
                raac_pssm_fs = extract_reduce(pos_matrix + neg_matrix, pos_aaid + neg_aaid, raaBook, raacode)
                raac_pssm_fs = extract_scale(raac_pssm_fs)
                method_id += '0'
            # RAAC-KPSSM
            if 1 in method:
                raac_kpssm_fs = ifeat.feature_kpssm(pos_matrix + neg_matrix, raaBook, raacode)
                raac_kpssm_fs = extract_scale(raac_kpssm_fs)
                method_id += '1'
            # RAAC-DTPSSM
            if 2 in method:
                raac_dtpssm_fs = ifeat.feature_dtpssm(pos_matrix + neg_matrix, raaBook, raacode)
                raac_dtpssm_fs = extract_scale(raac_dtpssm_fs)
                method_id += '2'
            # RAAC-SW
            if 3 in method:
                raac_sw_fs = ifeat.feature_sw(pos_matrix + neg_matrix, pos_aaid + neg_aaid, raaBook, raacode)
                raac_sw_fs = extract_scale(raac_sw_fs)
                method_id += '3'
            # RAAC-KMER
            if 4 in method:
                raac_kmer_fs = ifeat.feature_kmer(pos_aaid + neg_aaid, raacode)
                raac_kmer_fs = extract_scale(raac_kmer_fs)
                method_id += '4'
            # OAAC
            if 5 in method:
                raac_oaac_fs = ifeat.feature_oaac(pos_aaid + neg_aaid, raacode)
                raac_oaac_fs = extract_scale(raac_oaac_fs)
                method_id += '5'
            # SAAC
            if 6 in method:
                raac_saac_fs = ifeat.feature_saac(pos_aaid + neg_aaid, raacode)
                raac_saac_fs = extract_scale(raac_saac_fs)
                method_id += '6'
            # combine and save
            pssm_path = extract_combine([raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs], pos_type + neg_type, out, reduce, raacode, method_id)
            return pssm_path
        else:
            raac_pssm_fs = extract_reduce_row(pos_matrix + neg_matrix, pos_aaid + neg_aaid, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_reduce_col(raac_pssm_fs, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_scale(raac_pssm_fs)
            raac_pssm_fs = extract_one(raac_pssm_fs)
            file = extract_save(raac_pssm_fs, pos_type + neg_type, out, '0000-t0s20', '&0')
            pssm_path.append(file)
            return pssm_path
    elif pos != None and neg == None:
        pos_matrix, pos_aaid, pos_type = extract_single(pos, 0)
        if reduce != False and raaBook != None:
            raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs = [], [], [], [], [], [], []
            raacode = iload.load_raac(os.path.join(raac_path, raaBook))
            method_id = '-'
            # RAAC-PSSM
            if 0 in method:
                raac_pssm_fs = extract_reduce(pos_matrix, pos_aaid, raaBook, raacode)
                raac_pssm_fs = extract_scale(raac_pssm_fs)
                method_id += '0'
            # RAAC-KPSSM
            if 1 in method:
                raac_kpssm_fs = ifeat.feature_kpssm(pos_matrix, raaBook, raacode)
                raac_kpssm_fs = extract_scale(raac_kpssm_fs)
                method_id += '1'
            # RAAC-DTPSSM
            if 2 in method:
                raac_dtpssm_fs = ifeat.feature_dtpssm(pos_matrix, raaBook, raacode)
                raac_dtpssm_fs = extract_scale(raac_dtpssm_fs)
                method_id += '2'
            # RAAC-SW
            if 3 in method:
                raac_sw_fs = ifeat.feature_sw(pos_matrix, pos_aaid, raaBook, raacode)
                raac_sw_fs = extract_scale(raac_sw_fs)
                method_id += '3'
            # RAAC-KMER
            if 4 in method:
                raac_kmer_fs = ifeat.feature_kmer(pos_aaid, raacode)
                raac_kmer_fs = extract_scale(raac_kmer_fs)
                method_id += '4'
            # OAAC
            if 5 in method:
                raac_oaac_fs = ifeat.feature_oaac(pos_aaid, raacode)
                raac_oaac_fs = extract_scale(raac_oaac_fs)
                method_id += '5'
            # SAAC
            if 6 in method:
                raac_saac_fs = ifeat.feature_saac(pos_aaid, raacode)
                raac_saac_fs = extract_scale(raac_saac_fs)
                method_id += '6'
            # combine and save
            pssm_path = extract_combine([raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs], pos_type, out, reduce, raacode, method_id)
            return pssm_path
        else:
            raac_pssm_fs = extract_reduce_row(pos_matrix, pos_aaid, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_reduce_col(raac_pssm_fs, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_scale(raac_pssm_fs)
            raac_pssm_fs = extract_one(raac_pssm_fs)
            file = extract_save(raac_pssm_fs, pos_type, out, '0000-t0s20', '&0')
            pssm_path.append(file)
            return pssm_path
    elif neg != None and pos == None:
        neg_matrix, neg_aaid, neg_type = extract_single(neg, 1)
        if reduce != False and raaBook != None:
            raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs = [], [], [], [], [], [], []
            raacode = iload.load_raac(os.path.join(raac_path, raaBook))
            method_id = '-'
            # RAAC-PSSM
            if 0 in method:
                raac_pssm_fs = extract_reduce(neg_matrix, neg_aaid, raaBook, raacode)
                raac_pssm_fs = extract_scale(raac_pssm_fs)
                method_id += '0'
            # RAAC-KPSSM
            if 1 in method:
                raac_kpssm_fs = ifeat.feature_kpssm(neg_matrix, raaBook, raacode)
                raac_kpssm_fs = extract_scale(raac_kpssm_fs)
                method_id += '1'
            # RAAC-DTPSSM
            if 2 in method:
                raac_dtpssm_fs = ifeat.feature_dtpssm(neg_matrix, raaBook, raacode)
                raac_dtpssm_fs = extract_scale(raac_dtpssm_fs)
                method_id += '2'
            # RAAC-SW
            if 3 in method:
                raac_sw_fs = ifeat.feature_sw(neg_matrix, neg_aaid, raaBook, raacode)
                raac_sw_fs = extract_scale(raac_sw_fs)
                method_id += '3'
            # RAAC-KMER
            if 4 in method:
                raac_kmer_fs = ifeat.feature_kmer(neg_aaid, raacode)
                raac_kmer_fs = extract_scale(raac_kmer_fs)
                method_id += '4'
            # OAAC
            if 5 in method:
                raac_oaac_fs = ifeat.feature_oaac(neg_aaid, raacode)
                raac_oaac_fs = extract_scale(raac_oaac_fs)
                method_id += '5'
            # SAAC
            if 6 in method:
                raac_saac_fs = ifeat.feature_saac(neg_aaid, raacode)
                raac_saac_fs = extract_scale(raac_saac_fs)
                method_id += '6'
            # combine and save
            pssm_path = extract_combine([raac_pssm_fs, raac_kpssm_fs, raac_dtpssm_fs, raac_sw_fs, raac_kmer_fs, raac_saac_fs, raac_oaac_fs], neg_type, out, reduce, raacode, method_id)
            return pssm_path
        else:
            raac_pssm_fs = extract_reduce_row(neg_matrix, neg_aaid, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_reduce_col(raac_pssm_fs, 'raaCODE', '0000-t0s20')
            raac_pssm_fs = extract_scale(raac_pssm_fs)
            raac_pssm_fs = extract_one(raac_pssm_fs)
            file = extract_save(raac_pssm_fs, neg_type, out, '0000-t0s20', '&0')
            pssm_path.append(file)
            return pssm_path
    else:
        return pssm_path
