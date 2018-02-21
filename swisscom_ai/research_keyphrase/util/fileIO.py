# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import codecs

codecs.register_error('replace_with_space', lambda e: (u' ', e.start + 1))


def write_string(s, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write(s)


def read_file(input_path):
    with open(input_path, 'r', errors='replace_with_space') as input_file:
        return input_file.read().strip()
