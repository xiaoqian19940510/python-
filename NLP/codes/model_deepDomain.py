import logging
import numpy
import theano
import os
import cPickle
import json
import shutil

logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self):
        self.floatX = theano.config.floatX
        # Parameters of the model
        self.params = []

    def save(self, filename):
        """
        Save the model to file `filename`
        """
        path_name = os.path.dirname(filename)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        fname = os.path.join(path_name, "model.statetxt")
        cfg_file = open(fname, 'w')
                
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)
        cfg_file.write("model = \'%s\'\n"%(self.state['model_name']+".npz"))
        
        
        fname = os.path.join(path_name, self.state["vocabulary"])
        with open(fname, 'wb') as f: cPickle.dump(self.state['ref_voca'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("vocabulary = \'%s\'\n"%(self.state["vocabulary"]))
        
        fname = os.path.join(path_name, self.state["label"])        
        with open(fname, 'wb') as f: cPickle.dump(self.state['label_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("label = \'%s\'\n"%(self.state["label"]))

        fname = os.path.join(path_name, self.state["domain"])        
        with open(fname, 'wb') as f: cPickle.dump(self.state['domain_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("domain = \'%s\'\n"%(self.state["domain"]))
        
        cfg_file.write("phraselist = {0}\n".format(self.state["phraselist"])) 

        fname = os.path.join(path_name, self.state["stacklist"])             
        with open(fname, 'wb') as f: cPickle.dump(self.state['stack_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("stacklist = \'%s\'\n"%(self.state["stacklist"]))

        smask = self.state["STATE_LIST"]
        fname = os.path.join(path_name, self.state["stackmask"])
        with open(fname, 'wb') as f: json.dump(smask, f)
        cfg_file.write("stackmask = \'%s\'\n"%(self.state["stackmask"]))



               
        dmask = {}
        for cxt in smask:
            dmask[cxt] = []
            for intent_id in smask[cxt]:
                dmask[cxt].append(intent_id.split("%")[0])                

            dmask[cxt] = list(set(dmask[cxt]))

        fname = os.path.join(path_name, self.state["domainmask"])             
        with open(fname, 'wb') as f: json.dump(dmask, f)
        cfg_file.write("domainmask = \'%s\'\n"%(self.state["domainmask"]))
                        
        cfg_file.write("max_x = %i\n"%(self.state['max_x']))          
        cfg_file.write("cdim = %i\n"%(self.state['cdim']))   
            
        cfg_file.write("nConv = %i\n"%(self.state['nConv'])) 
        cfg_file.write("featuremap = {0}\n".format(self.state['featuremap'])) 
        cfg_file.write("nkerns = {0}\n".format(self.state['nkerns'])) 
        cfg_file.write("nHidden = %i\n"%(self.state['nHidden'])) 
        cfg_file.write("hidden = {0}\n".format(self.state['hidden'])) 
        cfg_file.write("seed = %i\n"%(self.state['seed'])) 
#add
        cfg_file.write("with_space = \'%s\'\n"%(self.state["with_space"]))
#added                                                        
        cfg_file.close()
    def save_final_config(self):

        """
        Save final model and parma on model_path`
        """

        fname = os.path.join(self.state["model_path"], "model.statetxt")
        cfg_file = open(fname, 'w')
        
        fname = os.path.join(self.state["model_path"], self.state["model_name"])
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(fname, **vals)
        cfg_file.write("model = \'%s\'\n"%(self.state["model_name"]+".npz"))
        
        fname = os.path.join(self.state["model_path"], self.state["vocabulary"])
        with open(fname, 'wb') as f: cPickle.dump(self.state['ref_voca'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("vocabulary = \'%s\'\n"%(self.state["vocabulary"]))
        
        fname = os.path.join(self.state["model_path"], self.state["label"])        
        with open(fname, 'wb') as f: cPickle.dump(self.state['label_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("label = \'%s\'\n"%(self.state["label"]))

        fname = os.path.join(self.state["model_path"], self.state["domain"])        
        with open(fname, 'wb') as f: cPickle.dump(self.state['domain_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("domain = \'%s\'\n"%(self.state["domain"]))
        
        cfg_file.write("phraselist = {0}\n".format(self.state["phraselist"]))
        for plist in self.state["phraselist"]:
            shutil.copy(os.path.join(self.state["data_path"], plist + ".plist"), self.state["model_path"])
        
        fname = os.path.join(self.state["model_path"], self.state["stacklist"])             
        with open(fname, 'wb') as f: cPickle.dump(self.state['stack_list'], f, protocol=cPickle.HIGHEST_PROTOCOL)
        cfg_file.write("stacklist = \'%s\'\n"%(self.state["stacklist"]))




           
        smask = self.state["STATE_LIST"]
        fname = os.path.join(self.state["model_path"], self.state["stackmask"])             
        with open(fname, 'wb') as f: json.dump(smask, f)
        cfg_file.write("stackmask = \'%s\'\n"%(self.state["stackmask"]))
#add (from)
        cfg_file.write("with_space = \'%s\'\n"%(self.state["with_space"]))
# to                   
        dmask = {}
        for cxt in smask:
            dmask[cxt] = []
            for intent_id in smask[cxt]:
                dmask[cxt].append(intent_id.split("%")[0])

            dmask[cxt] = list(set(dmask[cxt]))
        fname = os.path.join(self.state["model_path"], self.state["domainmask"])             
        with open(fname, 'wb') as f: json.dump(dmask, f)
        cfg_file.write("domainmask = \'%s\'\n"%(self.state["domainmask"]))
                           
        cfg_file.write("max_x = %i\n"%(self.state['max_x']))          
        cfg_file.write("cdim = %i\n"%(self.state['cdim']))   
            
        cfg_file.write("nConv = %i\n"%(self.state['nConv'])) 
        cfg_file.write("featuremap = {0}\n".format(self.state['featuremap'])) 
        cfg_file.write("nkerns = {0}\n".format(self.state['nkerns'])) 
        cfg_file.write("nHidden = %i\n"%(self.state['nHidden'])) 
        cfg_file.write("hidden = {0}\n".format(self.state['hidden'])) 
        cfg_file.write("seed = %i\n"%(self.state['seed'])) 
                                                        
        cfg_file.close()
                
    def load(self, filename):
        """
        Load the model.
        """
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                logger.debug('Loading {0} of {1}'.format(p.name, p.get_value(borrow=True).shape))
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception('Shape mismatch: {0} != {1} for {2}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(numpy.float32(vals[p.name]))
            else:
                logger.error('No parameter {0} given: default initialization used'.format(p.name))
                unknown = set(vals.keys())
                for name in self.params: unknown - name
                if len(unknown):
                    logger.error('Unknown parameters {0} given'.format(unknown))
    
    def load_no_echo(self, filename):
        """
        Load the model without echo.
        """
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception('Shape mismatch: {0} != {1} for {2}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(numpy.float32(vals[p.name]))
            else:
                logger.error('No parameter {0} given: default initialization used'.format(p.name))
 #               unknown = set(vals.keys())
 #               for name in self.params: unknown - name
                unknown = set(vals.keys()) - {p.name for p in self.params}
                raise Exception('loading value name is mismatched: for deploying')

                if len(unknown):
                    logger.error('Unknown parameters {0} given'.format(unknown))                    
