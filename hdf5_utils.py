# import the necessary packages
import h5py
import os
import torch
import time 
import collections


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


    database = h5py.File(f_name, 'w')

    write_attrs(database, param['dataset'] ) 
    
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
    database = h5py.File(file_name, 'r')                  # open db


    param = load_attrs(database)                        # print attrs if debug

    train = (database.get('train/input').value, database.get('train/truth').value )
    val = (database.get('val/input').value, database.get('val/truth').value )
    test = (database.get('test/input').value, database.get('test/truth').value )

    database.close()
    return train, val, test, param


"""

    Helper to store pytorch model

"""


def load_torch(f_name, MODEL_class):
    
    db = h5py.File(f_name, 'r')
    
    params = load_attrs(db)
    db_params = load_attrs(db['dataset'])
    
    mod = db['model']
    mod_params = load_attrs(mod)
    
    params['dataset'] = db_params
    params['input_s'] = mod_params['input_s'],
    params['hidden_s'] = mod_params['hidden_s'],
    params['output_s'] = mod_params['output_s'],
    params['device'] = torch.device(params['device'])
    
    
    model = MODEL_class(mod_params['input_s'], mod_params['hidden_s'], mod_params['output_s'])
    
    
    stat = collections.OrderedDict() 
    for l in mod_params['layers']:
        layer_param = load_attrs(mod[l])
        stat[l] = torch.from_numpy(mod.get(l).value,).to(layer_param['device'])
    
    db.close()
    
    model.load_state_dict(stat)
    
    
    
    return model, params


def save_torch(model, optimizer, f_name, param, crator="zehndiii"):
    #TODO Apend mode
    #TODO no overwrite
    # TODO support comments
    
    db = h5py.File(f_name, 'w')
    
    # write general settings
    write_attrs(db, {'creator':"zehndiii",       # write general attributes
                      'date':time.ctime(),
                      'epochs':param['epochs'],
                      'batch_size':param['batch_size'],
                #'optimizer':str(type(optimizer)),
                      'lr':param['lr'],
                      'decay':param['decay'],
                      'decay_step':param['decay_step'],
                      'device':str(param['device'])
                       })
    
    

    # write dataset settings
    dataset = db.create_group('/dataset')
    
    write_attrs(dataset, param['dataset'])
    """
    {'creator':creator,       # write general attributes
                      'date':date,
                      'neighbors':number_nei,
                      'augmentation':str(augmentation),
                      'shuffle':shuffle,
                      'truth_with_vel':truth_with_vel,
                      'mode':mode,
                      'fps':FPS
                     }
    """
    
    # write model and settings
    mod = db.create_group('/model')
   
    stat = model.state_dict()
    
    layers = []
    
    for k, v in stat.items():
        
        layers += [k]
        
        dat = v.cpu().detach().numpy()
        ds = mod.create_dataset( name=k,shape=v.shape, 
                                         dtype=dat.dtype,
                                         data=dat, compression="lzf" )

        write_attrs(ds, {'device':str(v.device.type),
                            'dtype':str(v.dtype),
                         })
    
    write_attrs(mod, {'input_s':param['input_s'],
                         'hidden_s':param['hidden_s'],
                         'output_s':param['output_s'],
                         'layers':layers,
                         })
    
    db.close()

    


