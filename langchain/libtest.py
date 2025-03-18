########################################################################
#
# $ python libtest.py input.txt output.txt --arg3 11111 -a 22222
#     arg1=input.txt
#     arg2=output.txt
#     arg3=11111
#     arg4=22222
# $ python libtest.py -h
#
import argparse

parser = argparse.ArgumentParser(
    description="test parser by mike"
)

parser.add_argument('arg1', help='this is help for arg1')
parser.add_argument('arg2', help='this is help for arg2')
parser.add_argument('--arg3', help='this is help for arg3')
parser.add_argument('-a', '--arg4')

args = parser.parse_args()

print('arg1='+args.arg1)
print('arg2='+args.arg2)
print('arg3='+args.arg3)
print('arg4='+args.arg4)

########################################################################
#
#
from settings import Settings

settings = Settings()
settings.readenv()

print("-------------end-------------")

