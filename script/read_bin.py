import struct
import sys

filename = sys.argv[1]

with open(filename, mode='rb') as f:
    print(filename, 'info')

    size = struct.unpack('<qqq', f.read(8*3))
    gc, = struct.unpack('<q', f.read(8))
    var_count, = struct.unpack('<q', f.read(8))
    var_dim = struct.unpack('<' + var_count*'q', f.read(var_count*8))
    var_name = []
    for v in range(var_count):
        char_count, = struct.unpack('<q', f.read(8))
        s = struct.unpack('<' + char_count*'c', f.read(char_count))
        s = str(b''.join(s))
        var_name.append(s)
    
    print('size =', size)
    print('gc =', gc)
    print('var count =', var_count)
    print('var dim =', var_dim)
    print('var name =', var_name)
    
