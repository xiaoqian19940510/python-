import logging
import argparse
import theano
import cnn_intent_deepintent
import os
import math
import time
import cPickle
import state_parser_deepintent as sp
from data_iterator import *
import random
import shutil
import codecs
from noaho import NoAho

logger = logging.getLogger(__name__)
start_time = str(time.time())
#...........................................................................................
def load_phrase_list(data_path, phrase_list):
    phrase_size = len(phrase_list)
    phrase_max = max(phrase_size, 1)
    pdict = []
    phrase_to_id = dict()
    for i in range(len(phrase_list)):
        phrase_to_id[phrase_list[i]] = i
        pdict.append(NoAho())
        with codecs.open(os.path.join(data_path, phrase_list[i] + ".plist"), "r", "utf8") as f:
            for line in f:
                pdict[i].add(line.strip().lower(), i)

    return pdict, phrase_to_id, phrase_size, phrase_max

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="the model directory", required=True)
    parser.add_argument("--test_file", type=str, help='test file name', required=True)
    parser.add_argument("--nbest_num", type=int, help='nbest num', default=1)
    parser.add_argument("--output_dir", type=str, help='output path', default="")


    args = parser.parse_args()
    args.test_file = sp.get_path_name(args.test_file)
    args.model_path = sp.get_path_name(args.model_path)
    return args

def writePerf(fp, ref, label, prob, strings, stacks, id_to_char, id_to_stack, start, maxl, utt_id, utt_level, utt_type):
    idx = start
    utt = ""

    while True:
        if strings[0][idx] == 0 or strings[0][idx] == 2: break
        if strings[0][idx] < 3 :
            idx += 1
            continue
        utt += id_to_char[strings[0][idx]]
        idx += 1
        if idx > maxl + start : break
    
    stack = id_to_stack[stacks]
    fp.write("%s\t%s\t%s(%.2f)\t%s\t( for WUT info [ utt_id : %s, utt_level : %s, utt_type : %s ] )\n" % (stack, ref, label, prob, utt, utt_id, utt_level, utt_type))
    
def main(args):
    state = sp.StateParse(os.path.join(args.model_path, "model.statetxt")).get_state()
    state["data_path"] = args.model_path
    state['bs'] = 100
    logging.basicConfig(level=getattr(logging, 'DEBUG'), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug('Start Time: {}'.format(start_time))
    tmp_path = ''
    if args.output_dir == "" :
        tmp_path = os.path.join(os.path.join(os.path.expanduser("~"), "tmp"), os.path.splitext(os.path.basename(args.test_file))[0])
    else :
        tmp_path = os.path.join(sp.get_path_name(args.output_dir),os.path.splitext(os.path.basename(args.test_file))[0])

    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)            
     
    perf_name = os.path.join(tmp_path, "perf")
    cmatrix_name = os.path.join(tmp_path, "cmatrix")
    intent_model = cnn_intent_deepintent.IntentParallelNet(state)
    
    model_name = os.path.join(state["data_path"], state['model'])
    if os.path.isfile(model_name):
        logger.debug('Loading existing model')
        intent_model.load(model_name)
    else:
        raise Exception("Must specify a valid model path")

    test_func = intent_model.build_test_only_function()
    
    if os.path.splitext(args.test_file)[1] == '.pkl':
        logger.debug('Loading testing data {}'.format(args.test_file))
        test_dataCNN = DataIteratorCNN(state['max_x'], state['max_l'], args.test_file ,state['bs'],state['padding'],state['psize']) 
    elif os.path.splitext(args.test_file)[1] == '.expand' or os.path.splitext(args.test_file)[1] == '.txt' :
        logger.debug('Converting testing data {}'.format(args.test_file))
        pdict, phrase_to_id, phrase_len, phrase_max  = load_phrase_list(state["data_path"], state["phraselist"])
        tmp_file = ConvertDataFormat(args.test_file, tmp_path, state["data_path"], intent_model.state, pdict, phrase_len, phrase_max)
        test_dataCNN = DataIteratorCNN(state['max_x'], state['max_l'], tmp_file ,state['bs'],state['padding'],state['psize']) 
    else:
        logger.debug('Unknown data expansion format test file {}'.format(args.test_file))
    test_dataCNN.data_set()
        
    id_to_label = dict()
    for key in state['label_list']:
        id_to_label[state['label_list'][key]]=key
    id_to_char = dict()
    for key in state['voca_list']:
        id_to_char[state['voca_list'][key]] = key 
    id_to_char[3] = "~"  
    id_to_stack = dict()
    for key in state['stack_list']:
        id_to_stack[state['stack_list'][key]] = key  

    stack_info = sp.parse_json(os.path.join(state["data_path"], state['stackmask']))
    if state['focused_on_rule_switching'].lower() == 'true' :
        base = state['max_l']
    else :
        base = state['max_l']/2    
    stackmask = np.ones((len(state['stack_list']), state['max_l']), 'float32') * -100
    for rid in stack_info['pseudo_root']:
        for i in range(len(state['stack_list'])):
            stackmask[i][state['label_list'][rid]] = 0
    for cxt in stack_info:
        if cxt != "pseudo_root":
            for rid in stack_info[cxt]: 
                if state['focused_on_rule_switching'].lower() == 'true' :
                    stackmask[state['stack_list'][cxt]][state['label_list'][rid]] = 0
                else :
                    stackmask[state['stack_list'][cxt]][state['label_list'][rid] + base] = 0
                    
    max_ver = len(state['vdict_list'])
    vtable = np.zeros((max_ver,state['max_l']), "float32")
    for i in range(max_ver):
        for j in range(base):
            vtable[i][j] = state['vtable_list'][i][j]
            if state['focused_on_rule_switching'].lower() == 'false' :
                vtable[i][j+base] = state['vtable_list'][i][j]
    vnum = np.zeros((state['max_l']), "int32")
    vlist = np.zeros((state['max_l'], max_ver), "int32")
    for rid in range(base):
        for vid in range(max_ver):
            if state['vtable_list'][vid][rid] == 0:
                vlist[rid][vnum[rid]] = vid
                vnum[rid] += 1        
                if state['focused_on_rule_switching'].lower() == 'false' :
                    vlist[rid+base][vnum[rid+base]] = vid
                    vnum[rid+base] += 1 
                 
                
    if state['focused_on_rule_switching'].lower() == 'true' :
        cmatrix = np.zeros((state['max_l'], state['max_l'])).astype('int32')
    else :
        cmatrix = np.zeros((state['max_l']/2, state['max_l']/2)).astype('int32')

    bcount = 0
    count = 0
    test_acc = 0
    nbest = np.zeros(args.nbest_num).astype('int32')
    
    logger.debug('Start batch test')
    fperf = codecs.open(perf_name, "w", 'utf-8')
    fcmatrix = codecs.open(cmatrix_name, "w", 'utf-8')
    while bcount * state['bs'] < test_dataCNN.size() :
        t_batch, bsize = test_dataCNN.next()
        #sm = sp.stack_label_mask(stackmask, t_batch[2], id_to_stack, state['label_list'], state['bs'], state['max_l'])
        sm = sp.train_version_stack_label_mask(stackmask, t_batch[2], id_to_stack, max_ver, vtable, vnum, vlist, t_batch[3], state['label_list'], id_to_label, state['bs'], state['max_l'])
        
        pred, smax, _, _ = test_func(np.asarray(t_batch[0]), np.asarray(t_batch[1]), np.asarray(t_batch[2]).astype('int32'), sm, 0)
        
        bcount += 1
        count += bsize
        offset = state['bs'] - bsize
        for i in range(bsize):
            idx = offset+i
            cmatrix[t_batch[3][idx]][pred[idx]%base] += 1
            if pred[idx]%base != t_batch[3][idx]:
                writePerf(fperf, id_to_label[t_batch[3][idx]], id_to_label[pred[idx]%base], smax[idx][pred[idx]], t_batch[0][idx], t_batch[2][idx], id_to_char, id_to_stack, state['padding'], state['max_x'],t_batch[5][idx],t_batch[6][idx],t_batch[7][idx])
            
            if args.nbest_num > 1:
                sidx = np.argsort(np.array(smax[idx]),)[::-1]
                for nth in range(args.nbest_num):
                    if sidx[nth] == t_batch[3][idx]:
                        nbest[nth] += 1
                        break        

    outs = np.zeros((base)).astype('int32')
    ins = np.zeros((base)).astype('int32')
    cm = np.zeros((state['max_l'], base)).astype('float32')
    prec = np.zeros((base)).astype('int32')
    recal = np.zeros((base)).astype('int32')
    for i in range(base):
        sums = 0
        for j in range(base):
            prec[j] += cmatrix[i][j]
            recal[i] += cmatrix[i][j]
            sums += cmatrix[i][j]
        if sums == 0: outs[i] = 1
        for j in range(base):
            if sums == 0: cm[i][j] = 0
            else:
                cm[i][j] = float(cmatrix[i][j])/sums
                if i != j and cm[i][j] > 0.001: ins[j] = 1
        if cm[i][i] > 0.999: outs[i] = 1
    for j in range(base):
        for i in range(base):
            if i != j and cm[i][j] > 0.001: ins[j] = 1

    for i in range(base):
        if outs[i] == 1 and ins[i] == 0: continue
        fcmatrix.write("{}\t".format(id_to_label[i]))
        for j in range(base):
            if outs[j] == 1 and ins[j] == 0: continue
            fcmatrix.write("{}\t".format(cm[i][j]))
        fcmatrix.write("\n")

    for i in range(base):
        test_acc += cmatrix[i][i]
        if recal[i] > 0 :
            print ("{} {} {} {}".format(id_to_label[i], recal[i], float(cmatrix[i][i])/recal[i], float(cmatrix[i][i])/prec[i]) )

    fperf.close()
    fcmatrix.close()
    if args.nbest_num <= 1:
        print('==== Test Accuracy   : {}  ({}/{})========='.format(float(test_acc)/count, test_acc, count))
    else:
        for i in range(args.nbest_num):
            if i > 0: nbest[i] += nbest[i-1]
            print('==== {}-best Accuracy: {}  ========='.format(i+1, float(nbest[i])/count))


if __name__ == "__main__":
    #assert(theano.config.floatX == 'float32')
    theano.config.floatX = 'float32'
    args = parse_args()
    main(args)
