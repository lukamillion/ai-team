# import the necessary packages
import h5py
import os
import torch
import time 
import collections
import numpy as np

class HDF5Error(Exception):
  pass    # Drop a custom error 


"""
    
    Basic manipulation and diagnostic of HDF5 files

"""

def fmt_fsize(num, suffix='B'):
    """
    str = fmt_fsize(num, suffix='B')
    
    Returns formated filesize in bytes
    """
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def write_attrs( db, D):
    """
    write_attrs( db, D)
    
    Writes a Attribute-Dictionary to a Database
    """
    for (key, value) in zip(list(D.keys()), list(D.values())):
        db.attrs[key] = value

def load_attrs( db ):
    """
    loads attributes dict from database
    """
    attr = {}
    for item in db.attrs.keys():
        attr[item] =  db.attrs[item]
    return attr


def show_attr(db, tree=""):
    """
    list all attributes asociated with a h5 object
    :param db: object to inspect
    :param tree: sufix
    """
    for item in db.attrs.keys():
        print(tree+'@'+item + ":", db.attrs[item])

def show_subdir(db, tree='|'):
    """
        list all subdirs and attributes asociated with a h5 object
        :param db: object to inspect
        :param tree: sufix
        """
    for name, obj in db.items():
        if type(obj) is h5py.Group:
            print("{tree}\n{tree}Group: {name} ".format(tree=tree,  name=name))
            show_attr(obj, tree=tree+'     ')
            show_subdir(obj, tree='|     '+tree)
        elif type(obj) is h5py.Dataset:
            print("{}-----Dataset: {:15} - Shape: {} {}".format(tree, name, obj.shape, obj.dtype))
            show_attr(obj, tree=tree+'          ')

def print_stats(file_name):
    """
    Plot a summary of the HDF5 Database
    """
    db = h5py.File(file_name, "r")
    print("File: " + file_name + db.name)
    
    f_size = os.path.getsize(file_name)
    print("|     Size: {}".format(fmt_fsize(f_size)))
    
    #plot all attributes
    show_attr(db, '|     ')
    
    show_subdir(db, '|')
       
    db.close()
    


"""

    Helpers To save and load training data

"""
def save_trainingdata(f_name, param, train, val, test ):
    """
      Save a dataset that conists of input/ truth pairs and a train/val/test split

      PARAM:
        f_name: filename to store to
        param:  dictionary with a entry 'dataset' which is a dictionary with all parameters that are relevant to the dataset
        train:  train split of the dataset
        val:    val split of the dataset
        test:   test split of the dataset
    """
    # crate database overwrite if it exists
    database = h5py.File(f_name, 'w')

    # write all the parametes of the dataset to the 
    write_attrs(database, param['dataset'] ) 
    
    # write each split in a sub folder
    train_h = database.create_group('/train')
    train_h.create_dataset( name='input',shape=train[0].shape, 
                                         dtype=train[0].dtype,
                                         data=train[0], compression="lzf" )
    train_h.create_dataset( name='truth',shape=train[1].shape, 
                                         dtype=train[1].dtype,
                                         data=train[1], compression="lzf" )
    val_h = database.create_group('/val')
    val_h.create_dataset( name='input',shape=val[0].shape, 
                                         dtype=val[0].dtype,
                                         data=val[0], compression="lzf" )
    val_h.create_dataset( name='truth',shape=val[1].shape, 
                                         dtype=val[1].dtype,
                                         data=val[1], compression="lzf" )
    test_h = database.create_group('/test')
    test_h.create_dataset( name='input',shape=test[0].shape, 
                                         dtype=test[0].dtype,
                                         data=test[0], compression="lzf" )
    test_h.create_dataset( name='truth',shape=test[1].shape, 
                                         dtype=test[1].dtype,
                                         data=test[1], compression="lzf" )

    database.close()
    


def load_trainingdata(file_name):
    """
        Load a dataset that conists of input/ truth pairs and a train/val/test split

        PARAM:
          f_name: filename to store to

        RETURN:
          train:  train split of the dataset
          val:    val split of the dataset
          test:   test split of the dataset
          param:  dictionary with a entry 'dataset' which is a dictionary with all parameters that are relevant to the dataset
    """
    database = h5py.File(file_name, 'r')                  # open db


    param = load_attrs(database)                        # print attrs if debug

    # we convert to np arrays for more conveniant use
    train = (np.array(database['train/input']), np.array(database['train/truth']))
    val = (np.array(database['val/input']), np.array(database['val/truth']))
    test = (np.array(database['test/input']), np.array(database['test/truth']))

    database.close()
    return train, val, test, param


"""

    Helper to store pytorch model

    we store the model ether in single mode or in scan mode.

    single mode: 
    
    /
    @ hyperparameters

    /Dataset/
      @ Dataset attributes
    /Model/
      @ hidden_s : array of size of hidden layers
      @ input_s : # number of input nodes
      @ output_s : # number of output nodes
      @ layers: OrderedDict of the layers
      Weights for each layer
      ...
    
    scan mode:  we use a a group for each sub model eg

    /
    @ mode: mulit_model
    @ models: list with models in stored in thes file

    /MODEL_1/
      @ hyperparameters

      /MODEL_1/Dataset/
        @ Dataset attributes
      /MODEL_1/Model/
        @ hidden_s : array of size of hidden layers
        @ input_s : # number of input nodes
        @ output_s : # number of output nodes
        @ layers: OrderedDict of the layers
        Weights for each layer
        ...

    /MODEL_2/
      @ hyperparameters

      /MODEL_2/Dataset/
        @ Dataset attributes
      /MODEL_2/Model/
        @ hidden_s : array of size of hidden layers
        @ input_s : # number of input nodes
        @ output_s : # number of output nodes
        @ layers: OrderedDict of the layers
        Weights for each layer
        ...
    
    ...


"""


def load_torch(f_name, MODEL_class, prefix='model_1'):
    """
        Load a torch model and initialize a instance of the sored model. load all settings along the model.
        
        PARAM:
          f_name:       file to read from
          MODEL_class:  Torch model class (nn.Module)
          prefix:       if multiple networks are sotred in the file select a model
        RETURN:

    """
    # open a database
    database = h5py.File(f_name, 'r')

    # load all parameters from the 
    params = load_attrs(database)

    if params['mode']=='multi_model':
      print("multi_model")
      db = database[prefix]
      params = load_attrs(db)
    else:
      db = database

    db_params = load_attrs(db['dataset'])
    
    mod = db['model']
    mod_params = load_attrs(mod)
    
    params['dataset'] = db_params
    params['input_s'] = mod_params['input_s'],
    params['hidden_s'] = mod_params['hidden_s'],
    params['output_s'] = mod_params['output_s'],
    params['device'] = torch.device(params['device'])
    
    print(mod_params)

    # create model from stored parameters
    model = MODEL_class(mod_params['input_s'], mod_params['hidden_s'], mod_params['output_s'])
    
    # read the layers keeping the order
    stat = collections.OrderedDict() 
    for l in mod_params['layers']:
        layer_param = load_attrs(mod[l])      # get the device settings for each tensor
        stat[l] = torch.from_numpy(mod.get(l).value,).to(layer_param['device'])   # initialize the model weight tensors
    
    database.close()
    
    # load weights to the model
    model.load_state_dict(stat)
    
    
    return model, params


def save_torch(model, optimizer, f_name, param, scan=False, prefix='', creator="zehndiii"):
    """
        Save a torch model along with  all settings of the experiment.
        
        PARAM:
          model:        used model of type (nn.Module)
          optimizer:    instance of the used optimizer
          f_name:       file to save to must not exist except we are in scan mode
          param:        dict where all parametes of the experiment are stored
          scan:         if true multiple model can be safed to the same file. 
          prefix:       if multiple networks are sotred in the file select a model
          creator:      keep track who performed experiment 
        RETURN:

    """
    
    # TODO support comments
    
    # we do not want to overide any model we catch all atempts to do so
    if os.path.isfile(f_name):
      if scan:
        database = h5py.File(f_name, 'a')

      else:
        raise HDF5Error( "Cannot write to file that already exists: {}".format(f_name) )
    else:
      database = h5py.File(f_name, 'w')
    

    if scan:
      # check for valid parameters
      if prefix=="":
        database.close()
        raise HDF5Error("Cannot write model with no name specified" )

      par = load_attrs(database, )

      if par == {}:
        par['mode'] = "multi_model"
        par['models'] = np.array(["dummy"], dtype=object)


      elif par['mode']!="multi_model":
        database.close()
        raise HDF5Error("You cannot append to a single_model HDF5 dump!")

      if prefix.encode("ascii", "ignore") in par['models']:
        database.close()
        raise HDF5Error("Cannot replace model that already exists: {}".format(prefix))

      # we append the model to the model list
     
      par['models'] = [n.encode("ascii", "ignore") for n in np.append(par['models'] ,  prefix)]

      write_attrs(database, par)

      db = database.create_group(prefix)


      
    else:
      # we do not need to store additional parameters for a singel model
      print("simpe")
      db = database
      

    # write general settings
    write_attrs(db, { 'mode': 'single_model',
                      'creator':creator,       # write general attributes
                      'date':time.ctime(),
                      'epochs':param['epochs'],
                      'batch_size':param['batch_size'],
                #'optimizer':str(type(optimizer)),
                      'lr':param['lr'],
                      'decay':param['decay'],
                      'decay_step':param['decay_step'],
                      'device':str(param['device']),
                      'msg':param['msg'],
                       })
    
    

    # write dataset settings
    dataset = db.create_group('dataset')
    
    write_attrs(dataset, param['dataset'])
    
    # write model and settings
    mod = db.create_group('model')
   
    stat = model.state_dict()
    
    layers = []
    
    # loop over layers, get the weights as numpy array and store them to a new dataset
    # write the tensor attributes to the datset
    for k, v in stat.items():
        
        layers += [k]
        
        dat = v.cpu().detach().numpy()
        ds = mod.create_dataset( name=k,shape=v.shape, 
                                         dtype=dat.dtype,
                                         data=dat, compression="lzf" )

        write_attrs(ds, {'device':str(v.device.type),
                            'dtype':str(v.dtype),
                         })
    
    # store the order of the layers
    write_attrs(mod, {'input_s':param['input_s'],
                         'hidden_s':param['hidden_s'],
                         'output_s':param['output_s'],
                         'layers':layers,
                         })
    
    database.close()

    


