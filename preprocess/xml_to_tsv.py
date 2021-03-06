"""
Script to convert the xml data of https://github.com/chridey/change-my-view-modes/ 
to the tsv required for AMPERSAND evaluation.

Format of tsv generated (with --numbered flag):
<sentence_no>    <sentence>    <O|C|P>

Format of tsv generated (without --numbered flag):
<sentence>    <O|C|P>
"""

import os, re
import bs4
from bs4 import BeautifulSoup
import argparse

def clean_text(text: str) -> str:
    for elem in ['.', ',','!',';', ':']:
        text = text.replace(elem, ' '+elem+' ')
    text = text.strip(' _\t\n')
    text = text.split('____')[0]                                                    #To remove footnotes
    text = text.strip(' _\t\n')
    text = re.sub(r'\(https?://\S+\)', '<url>', text)                               #To remove URLs
    text = re.sub(r'&gt;.*(?!(\n+))$', '', text)                                    #To remove quotes at last.
    text = re.sub(r'\n', ' ', text)
    text = text.rstrip(' _\n\t')
    text = re.sub(r'\n', ' ', text)
    text = text.lower()
    return text

def refine_xml(xml_string):
        xml_string = re.sub(r'<claim [^>]*>', r'<claim>', xml_string)
        xml_string = re.sub(r'<premise [^>]*>', r'<premise>', xml_string)
        return xml_string

def build_tsv(parsed_xml):
    global str_to_write

    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        for elem in post.contents:
            elem = str(elem)
            if elem.startswith('<claim>'):
                elem = clean_text(elem[7:-8])
                str_to_write.append(str(len(str_to_write))+'\t'+elem+'\t'+'C')
            elif elem.startswith('<premise>'):
                elem = clean_text(elem[9:-10])
                str_to_write.append(str(len(str_to_write))+'\t'+elem+'\t'+'P')
            else:
                elem = clean_text(elem)
                if len(elem)<=1:
                    continue
                str_to_write.append(str(len(str_to_write))+'\t'+elem+'\t'+'O')

str_to_write = []
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Folder having the .xml files of https://github.com/chridey/change-my-view-modes/ data.')
    parser.add_argument('--write_file', type=str, required=True, help='Filename where the script should write the data in tsv format.')
    parser.add_argument('--numbered', action='store_true', help='If this flag is provided, the sentence numbers are omitted.')
    parser.add_argument('--append', action='store_true', help='If this flag is provided, the new data is appended to write file.')
    args = parser.parse_args()
    
    for f in os.listdir(args.folder):
        filename = os.path.join(args.folder, f)
        if os.path.isfile(filename) and filename.endswith('.xml'):
            with open(filename, 'r') as g:
                xml_str = g.read()
            parsed_xml = BeautifulSoup(refine_xml(xml_str), "xml")        
            build_tsv(parsed_xml)    

    with open(args.write_file, 'a' if args.append else 'w') as f:
        for elem in str_to_write:
            if not args.numbered:
                elem = '\t'.join(elem.split('\t')[1:])
            f.write(elem+'\n' if not elem.endswith('\n') else elem)