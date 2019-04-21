import os

def create_client(database_url = 'mongodb-simprod.icecube.wisc.edu',
                  dbuser = 'DBadmin',
                  password_path = os.path.expandvars('$HOME/.mongo')):

    import logging
    from os import environ
    from os.path import join
    from os.path import exists
    
    try:
        from pymongo import MongoClient
    except ImportError:
        logging.critical("PyMongo not installed.")

    if not exists(password_path):
        logging.critical("Password file '%s' not found." % password_path)
        
    f = open(password_path)
    uri = "mongodb://%s:%s@%s" % (dbuser, f.readline().strip(), database_url)
    f.close()
    
    client = MongoClient(uri)

    return client
    

