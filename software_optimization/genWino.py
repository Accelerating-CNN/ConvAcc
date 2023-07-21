import os,sys
sys.path.append("wincnn/")
from wincnn import *
from sympy import Rational


l = [
        [(0,1,-1),2,3],
        [(0,1,-1,Rational(-2,1),Rational(2,1)),4,3],
        [(0,1,-1,-2,2,Rational(1,2),-Rational(1,2)),4,5],
        [(0,1,-1,-2,2,Rational(1,2),-Rational(1,2),Rational(4,1),-Rational(4,1)),4,7],
        [(0,1,-1,-2,2,Rational(1,2),-Rational(1,2),Rational(4,1),-Rational(4,1),
          Rational(1,4),-Rational(1,4),Rational(8,1),-Rational(8,1)),4,11]
        ]
#generate header just for testing
#this is for different size of input and output
#possible to generate different size of input and output
#to go win.cnn
generateCHeader(l,"src/winograd.h")

