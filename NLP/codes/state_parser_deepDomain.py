import csv
import os
import re,sys
import codecs
import json
import argparse
import shutil
import numpy as np
import random
from theano.compile.io import Out

""" ============== generic functions =============="""
def get_path_name(input_path):
    """ Resolve path with environment variable such as $RESOURCES
        inp : path with environment variable
        out : absolute path
        pasted by sang-ho lee, 20160620, to resolve $RESOURCES in json file
    """
    
    inp = input_path.strip('/')
    out = ''
    ph = inp.split('/')
    for i in range(len(ph)):
        if ph[i][0] == '$':
            out += os.environ[ph[i][1:]]
        else:
            out += ph[i]
        if i < len(ph) -1 :
            out += '/'
    
    if input_path[0] == '/': return '/'+ out 
    else: return out

# Regular expression for comments

comment_re = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)
def parse_json(filename):
    """ Parse a JSON file
        First remove comments and then use the json module package
        Comments look like :
            // ...
        or
            /*
            ...
            */
    """
    if os.path.isfile(filename) == True:
        with open(filename) as f:
            content = ''.join(f.readlines())

            ## Looking for comments
            match = comment_re.search(content)
            while match:
                # single line comment
                content = content[:match.start()] + content[match.end():]
                match = comment_re.search(content)

        # Return json file
        return json.loads(content)
    else:
        print "ERROR: Cannot open " + filename + "!!"
        sys.exit()
        
def cleanupDir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)      
    
""" ================================================"""
##added for weighted sampling
def weightedpick(d):
    r = random.uniform(0, sum(d.itervalues()))
    s = 0.0
    for k,w in d.iteritems():
        s += w
        if r < s: return k
    return k



def stack_label_mask(bitmask, stack, bs, max_l):
    sm = np.zeros((bs, max_l), 'float32')
    
    for i in range(bs):
        sm[i] = sm[i] + bitmask[stack[i]]
    return sm

def stack_label_mask_neg(stackmask, stack, id_to_stack, label_list, bs, max_l):

    sm = np.zeros((bs, max_l), 'float32')
    
    for i in range(bs):
        for ids in stackmask[id_to_stack[stack[i]]]:
            if "followup" in ids: 
                sm[i][label_list[ids]] = -1
    return sm

def stack_label_reassign(stackmask, stacks, labels, domain, stack_to_id, id_to_stack, label_to_id, id_to_label, domain_to_id, bs ,phoenix_rule, total_state):
    num_stack = len(stack_to_id)
    new_stacks =  []
    new_dom = []


    for i in range(bs):
        sid = -1
        if stacks[i] == stack_to_id["pseudo_root"]:
            if id_to_label[labels[i]] in phoenix_rule:   
                if random.random() < 0.5:
                   if random.random() < 0.5:
                       sid = random.randrange(0,num_stack)
                   else:
                       sid = weightedpick(total_state)

            else:
                if random.random() < 0.3:
                   if random.random() < 0.5:
                       sid = random.randrange(0,num_stack)
                   else:
                       sid = weightedpick(total_state)
            if sid != -1:
                if sid in id_to_stack:
                    new_stacks.append(sid)
                    if id_to_stack[sid] != "pseudo_root" and id_to_label[labels[i]] in stackmask[id_to_stack[sid]]:
                        if id_to_label[labels[i]].split("%")[0] in domain_to_id:
                            new_dom.append(domain_to_id[id_to_label[labels[i]].split("%")[0]])
                        else:
                            new_dom.append(domain_to_id['#sports'])
                    else:
                        new_dom.append(domain[i])
                else:
                    new_stacks.append(sid)
                    new_dom.append(domain[i])                    
            else:
                new_stacks.append(stacks[i])
                new_dom.append(domain[i])                    


        else:
            new_stacks.append(stacks[i])
            if id_to_label[labels[i]] in stackmask[id_to_stack[stacks[i]]]:
                new_dom.append(domain[i]+len(domain_to_id))
            else:
                new_dom.append(domain[i])
            
    return new_stacks, new_dom

def stack_label_matrix(stackmask, stack, id_to_stack, label_list, bs, max_l):
    stack_matrix = np.zeros((bs, max_l), "float32")
    
    for i in range(bs):
        for id in stackmask[id_to_stack[stack[i]]]:
            stack_matrix[i][label_list[id]] = 1    

    return stack_matrix

class ConfigParse():
    def __init__(self, config_file):
        cfg = parse_json(get_path_name(config_file))
        self.state = cfg["MODEL_CONFIG"]
        self.state["domain_spec"] = get_path_name(cfg["DOMAIN_SPEC"])
        self.state["model_path"] = get_path_name(cfg["MODEL_OUT_PATH"])
        self.state["data_path"] = get_path_name(cfg["DATA_OUT_PATH"])
        self.state["model_tmp"] = get_path_name(cfg["MODEL_TMP_DIR"])
        if cfg.has_key('MULTI_GPU_WORKER'):
            wid = 0
            workers = []
            for worker in cfg['MULTI_GPU_WORKER']:
                ip, gpus, uid = worker
                for gpu in gpus:
                    workers.append([wid, ip, gpu, uid])
                    wid += 1                
            self.state["workers"] = workers
        
        if not os.path.exists(self.state["model_path"]):
            os.makedirs(self.state["model_path"])   
        if not os.path.exists(self.state["model_tmp"]):
            os.makedirs(self.state["model_tmp"])

    def get_state(self):
        return self.state
        
def comment_remover(iterator):
    for line in iterator:
        if not line.strip():
            continue
        if line.strip()[0] == '#':
            continue
        yield line

class StateParse():            
    def __init__(self, state_file):
        self.state = {}
        with open(state_file,'rb') as f:
            state_reader = csv.reader(comment_remover(f), delimiter='=')
            for key, val in state_reader:
                self.state[key.strip()] = eval(val.strip())
    
    def get_state(self):
        return self.state
    

if __name__ == '__main__':
    print "test"
    
