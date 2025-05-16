#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   grep.py
@Time    :   2024/04/18 17:28:29
@Author  :   George Trenins
@Desc    :   Pythonic emulation of `grep`
'''


from __future__ import print_function, division, absolute_import
import re
from typing import Optional

def grep(
    pattern : str,
    file_path : str,
    head : Optional[int] = 2147483648
) -> list[str]:
    pattern_match = re.compile(pattern)
    matches = []
    nmatch = 0

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if pattern_match.search(line):
                matches.append(line)
                nmatch += 1
                if nmatch == head: 
                    break
    return matches


def grep_context(
        pattern : str,
        file_path : str,
        before : Optional[int] = 2,
        after : Optional[int] = 2,
        head : Optional[int] = 2147483648) -> list[str]:
    """
    Args:
        pattern (str): regex patter
        file_path (str): name of file
        before (int, optional): number of context lines to return before the match. Defaults to 2.
        after (int, optional): context lines after the match. Defaults to 2.
        head (int, optional): only look for the first `head` occurences of the match
    """

    pattern_match = re.compile(pattern)
    context_buffer = []
    assert before >= 0, f'number of lines must be positive, instead {before=}'
    assert after >= 0, f'number of lines must be positive, instead {after=}'
    
    matches = []
    nmatch = 0

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # fill the buffer
            if i <= after + before:
                context_buffer.append(line)
            else:
                midline = context_buffer[before]
                if pattern_match.search(midline):
                    matches.append(''.join(context_buffer))
                    nmatch += 1
                    if nmatch == head:
                        return matches
                # Add the line to the context buffer
                context_buffer.append(line)
                context_buffer.pop(0)
        # finish off the buffer
        l = len(context_buffer)
        for _ in range(before + after + 1):
            midline = context_buffer[min(before, l-1)]
            if pattern_match.search(midline):
                matches.append(''.join(context_buffer))
            context_buffer.pop(0)
            l -= 1
    return matches


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern')
    parser.add_argument('filename')
    parser.add_argument('-B', type=int, default=0)
    parser.add_argument('-A', type=int, default=0)
    args = parser.parse_args()
    matches = grep_context(args.pattern, args.filename, before=args.B, after=args.A)
    print('--\n'.join(matches), end='')