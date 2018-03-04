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

logger = logging.getLogger(__name__)
start_time = str(time.time())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, help="trining paramameter configuration", default = "$RESOURCES/$lang/$proj/config/train/intent_train_config.json")
    parser.add_argument("--index_path", type=str, help="Data index file path", default="")
    args = parser.parse_args()
    return args
       
def save(model, note=''):
 
    if note: note = '_'+note 
    model_file = os.path.join(model.state['model_tmp'], model.state['model_name']+note)
    model.save(model_file)
    
def main(args):
    state = sp.ConfigParse(args.config_file).get_state()

    logging.basicConfig(level=getattr(logging, 'DEBUG'), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug('Start Time: {}'.format(start_time))
    
    intent_model = cnn_intent_deepintent.IntentParallelNet(state)
    train_func = intent_model.build_train_function()
    test_func = intent_model.build_test_function()
   
    if state.has_key('weights'):
        if os.path.isfile(state['weights']):
            logger.debug('Loading existing model_deepintent')
            intent_model.load(state['weights'])
        else:
            raise Exception("Must specify a valid model_deepintent path")
 
    save(intent_model, "")
    logger.debug('Loading training data iterator')
    train_dataCNN = DataIteratorCNN(state['max_x'], state['max_l'], os.path.join(state["data_path"], state['train_data']), state['bs'],state['padding'],state['psize'], args.index_path)
    if state['val_interval'] > 0:
        val_dataCNN = DataIteratorCNN(state['max_x'], state['max_l'], os.path.join(state["data_path"], state['val_data']),state['bs'],state['padding'],state['psize'], args.index_path)   
    
    logger.debug('Start training iterations.')
    batches = math.ceil(float(train_dataCNN.size())/state['bs'])
    cost_sum = 0.
    step = 0
    epoch = 0
    total_e = 0
    random.seed()
    output_cost = 0
    error = 0
    pstep = train_dataCNN.size()/20
    pstep = int(pstep - pstep % state['bs'])
    while epoch < state['epochs']:
        batch, bsize = train_dataCNN.next()
        
        if bsize == state['bs']:
            output_cost, error= train_func(np.asarray(batch[0]), np.asarray(batch[1]), np.asarray(batch[2]).astype('float32'), np.asarray(batch[3]).astype('int32'), 1)
        
        if (step+1 % batches == 0) or ((step*state['bs'] % pstep) < state['bs']):
            logger.debug('Training: Epoch {} / Step {}: {} / {} '.format(epoch+1, step+1, output_cost, error ))
            
        cost_sum += output_cost
        total_e += error
        step += 1

        # computing cost per epoch
        if step % batches == 0:
            epoch += 1
            logger.debug("\t>> Epoch {} Avg. Cost: {}, Avg. error: {}".format(epoch, cost_sum/batches, total_e/batches))
            total_e = 0
            
            if epoch % state['save_epochs'] == 0: 
                save(intent_model, str(epoch)) 
                
            # evaluation
            if state['val_interval'] > 0: 
                #held out
                eval_errors = 0
                bcount = 0
                count = 0
                while bcount * state['bs'] < val_dataCNN.size() :
                    e_batch, bsize = val_dataCNN.next()
                    
                    pred, errors = test_func(np.asarray(e_batch[0]), np.asarray(e_batch[1]), np.asarray(e_batch[2]).astype('float32'), np.asarray(e_batch[3]).astype('int32'), 0)
                    bcount += 1
                    count += bsize
                    for i in range(bsize):
                        if pred[i] != e_batch[3][i]:
                            eval_errors += 1
    
                logger.debug('\t>> Held out Eval Errors: Epoch {}  Errors {} '.format(epoch, float(eval_errors)/count ))
                 
            cost_sum = 0.
            
    #save(intent_model, "")
    
    intent_model.save_final_config()

    return 0
                
if __name__ == "__main__":
    theano.config.floatX = 'float32'
    #assert(theano.config.floatX == 'float32')
    
    args = parse_args()
    main(args)
