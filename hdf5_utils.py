# import the necessary packages
import h5py
import os


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
    

