missClassificationCorrection = \
    {1: 2, 3: 10, 4: 11, 5: 12, 6: 10, 7: 13, 8: 12, 9: 15, 143: 174, 25: 47, 26: 51,
     27: 49, 28: 51, 29: 57, 30: 53, 31: 59, 32: 55, 33: 62, 34: 58, 35: 65,
     36: 63, 37: 66, 38: 65, 39: 67, 40: 63, 41: 64, 42: 69, 44: 71, 45: 72,
     46: 74, 157: 189, 158: 188, 183: 191, 184: 192, 185: 193, 186: 194, 159: 190,
     75: 83, 77: 84, 79: 87, 173: 176, 156: 187, 99: 123, 100: 127, 101: 132,
     102: 136, 103: 124, 104: 128, 105: 131, 106: 135, 107: 139, 108: 140, 168: 175}


bravaisLattice2ibrav = {'cubic P':1,'cubic F':2,'cubic I':3,'hexagonal P':4,'trigonal P':4,
                            'trigonal R':5,'tetragonal P':6,'tetragonal I':7,
                            'orthorhombic P':8,'orthorhombic A':-9,'orthorhombic B':-9,
                            'orthorhombic C':-9,'orthorhombic F':10,'orthorhombic I':11,
                            'monoclinic P':-12,'monoclinic A':13,'monoclinic B':13,
                            'monoclinic C':13, 'triclinic P':14}

SG2BravaisLattice = \
    {1: 'triclinic P', 2: 'triclinic P', 3: 'monoclinic P', 4: 'monoclinic P',
     5: 'monoclinic C', 6: 'monoclinic P', 7: 'monoclinic P', 8: 'monoclinic C',
     9: 'monoclinic C', 10: 'monoclinic P', 11: 'monoclinic P', 12: 'monoclinic C',
     13: 'monoclinic P', 14: 'monoclinic P', 15: 'monoclinic C', 16: 'orthorhombic P',
     17: 'orthorhombic P', 18: 'orthorhombic P', 19: 'orthorhombic P',
     20: 'orthorhombic C', 21: 'orthorhombic C', 22: 'orthorhombic F',
     23: 'orthorhombic I', 24: 'orthorhombic I', 25: 'orthorhombic P',
     26: 'orthorhombic P', 27: 'orthorhombic P', 28: 'orthorhombic P',
     29: 'orthorhombic P', 30: 'orthorhombic P', 31: 'orthorhombic P',
     32: 'orthorhombic P', 33: 'orthorhombic P', 34: 'orthorhombic P',
     35: 'orthorhombic C', 36: 'orthorhombic C', 37: 'orthorhombic C',
     38: 'orthorhombic A', 39: 'orthorhombic A', 40: 'orthorhombic A',
     41: 'orthorhombic A', 42: 'orthorhombic F', 43: 'orthorhombic F',
     44: 'orthorhombic I', 45: 'orthorhombic I', 46: 'orthorhombic I',
     47: 'orthorhombic P', 48: 'orthorhombic P', 49: 'orthorhombic P',
     50: 'orthorhombic P', 51: 'orthorhombic P', 52: 'orthorhombic P',
     53: 'orthorhombic P', 54: 'orthorhombic P', 55: 'orthorhombic P',
     56: 'orthorhombic P', 57: 'orthorhombic P', 58: 'orthorhombic P',
     59: 'orthorhombic P', 60: 'orthorhombic P', 61: 'orthorhombic P',
     62: 'orthorhombic P', 63: 'orthorhombic C', 64: 'orthorhombic C',
     65: 'orthorhombic C', 66: 'orthorhombic C', 67: 'orthorhombic C',
     68: 'orthorhombic C', 69: 'orthorhombic F', 70: 'orthorhombic F',
     71: 'orthorhombic I', 72: 'orthorhombic I', 73: 'orthorhombic I',
     74: 'orthorhombic I', 75: 'tetragonal P', 76: 'tetragonal P', 77:
         'tetragonal P', 78: 'tetragonal P', 79: 'tetragonal I', 80: 'tetragonal I',
     81: 'tetragonal P', 82: 'tetragonal I', 83: 'tetragonal P', 84: 'tetragonal P',
     85: 'tetragonal P', 86: 'tetragonal P', 87: 'tetragonal I', 88: 'tetragonal I',
     89: 'tetragonal P', 90: 'tetragonal P', 91: 'tetragonal P', 92: 'tetragonal P',
     93: 'tetragonal P', 94: 'tetragonal P', 95: 'tetragonal P', 96: 'tetragonal P',
     97: 'tetragonal I', 98: 'tetragonal I', 99: 'tetragonal P', 100: 'tetragonal P',
     101: 'tetragonal P', 102: 'tetragonal P', 103: 'tetragonal P',
     104: 'tetragonal P', 105: 'tetragonal P', 106: 'tetragonal P', 107:
         'tetragonal I', 108: 'tetragonal I', 109: 'tetragonal I',
     110: 'tetragonal I', 111: 'tetragonal P', 112: 'tetragonal P', 113: 'tetragonal P',
     114: 'tetragonal P', 115: 'tetragonal P', 116: 'tetragonal P', 117: 'tetragonal P',
     118: 'tetragonal P', 119: 'tetragonal I', 120: 'tetragonal I', 121: 'tetragonal I',
     122: 'tetragonal I', 123: 'tetragonal P', 124: 'tetragonal P', 125: 'tetragonal P',
     126: 'tetragonal P', 127: 'tetragonal P', 128: 'tetragonal P', 129: 'tetragonal P',
     130: 'tetragonal P', 131: 'tetragonal P', 132: 'tetragonal P', 133: 'tetragonal P',
     134: 'tetragonal P', 135: 'tetragonal P', 136: 'tetragonal P', 137: 'tetragonal P',
     138: 'tetragonal P', 139: 'tetragonal I', 140: 'tetragonal I', 141: 'tetragonal I',
     142: 'tetragonal I', 143: 'trigonal P', 144: 'trigonal P', 145: 'trigonal P',
     146: 'trigonal R', 147: 'trigonal P', 148: 'trigonal R', 149: 'trigonal P',
     150: 'trigonal P', 151: 'trigonal P', 152: 'trigonal P', 153: 'trigonal P',
     154: 'trigonal P', 155: 'trigonal R', 156: 'trigonal P', 157: 'trigonal P',
     158: 'trigonal P', 159: 'trigonal P', 160: 'trigonal R', 161: 'trigonal R',
     162: 'trigonal P', 163: 'trigonal P', 164: 'trigonal P', 165: 'trigonal P',
     166: 'trigonal R', 167: 'trigonal R', 168: 'hexagonal P', 169: 'hexagonal P',
     170: 'hexagonal P', 171: 'hexagonal P', 172: 'hexagonal P', 173: 'hexagonal P',
     174: 'hexagonal P', 175: 'hexagonal P', 176: 'hexagonal P', 177: 'hexagonal P',
     178: 'hexagonal P', 179: 'hexagonal P', 180: 'hexagonal P', 181: 'hexagonal P',
     182: 'hexagonal P', 183: 'hexagonal P', 184: 'hexagonal P', 185: 'hexagonal P',
     186: 'hexagonal P', 187: 'hexagonal P', 188: 'hexagonal P', 189: 'hexagonal P',
     190: 'hexagonal P', 191: 'hexagonal P', 192: 'hexagonal P', 193: 'hexagonal P',
     194: 'hexagonal P', 195: 'cubic P', 196: 'cubic F', 197: 'cubic I',
     198: 'cubic P', 199: 'cubic I', 200: 'cubic P', 201: 'cubic P', 202: 'cubic F',
     203: 'cubic F', 204: 'cubic I', 205: 'cubic P', 206: 'cubic I', 207: 'cubic P',
     208: 'cubic P', 209: 'cubic F', 210: 'cubic F', 211: 'cubic I', 212: 'cubic P',
     213: 'cubic P', 214: 'cubic I', 215: 'cubic P', 216: 'cubic F', 217: 'cubic I',
     218: 'cubic P', 219: 'cubic F', 220: 'cubic I', 221: 'cubic P', 222: 'cubic P',
     223: 'cubic P', 224: 'cubic P', 225: 'cubic F', 226: 'cubic F', 227: 'cubic F',
     228: 'cubic F', 229: 'cubic I', 230: 'cubic I'}