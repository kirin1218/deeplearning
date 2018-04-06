# -*- coding: utf-8 -*-
import sys
import os
import hashlib
import zlib

def makeblobdata(data):
    content = data
    header = 'blob {}\0'.format(len(content))
    store = header + content
    print('data=' + store )
    
    return store.encode('sjis')

def makehash( data ):
    byte_data = makeblobdata(data)    
    
    h = hashlib.new('sha1')
    h.update(byte_data)

    return h.hexdigest()

def compressdata( data ):
    byte_data = makeblobdata(data)    

    return zlib.compress(byte_data)

    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv
        print( args[1] )
        hashdata = makehash(args[1])
        print( 'hashdata= ' + str(hashdata) )
        compdata = compressdata( args[1])
        print( 'compdata= ' + str(compdata) )
    else:
        print( 'invalid paramater' )


