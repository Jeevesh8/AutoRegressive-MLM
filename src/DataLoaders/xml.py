import os, re
from bs4 import BeautifulSoup

class load_xml_data:
    def __init__(self, config, split='train/'):
        self.config = config
        self.split = split

    def clean_text(self, text):

        text = text.strip(' _\t\n')
        text = text.split('____')[0]                                                    #To remove footnotes
        text = text.strip(' _\t\n')
        text = re.sub(r'\(https?://\S+\)', '<url>', text)                               #To remove URLs
        text = re.sub(r'&gt;.*(?!(\n+))$', '', text)                                    #To remove quotes at last.
        text = re.sub(r'&gt;(.*)\n', '<startq> \g<1> <endq>', text)                     #To add start quote, end quote tags
        text = re.sub(r'\n', ' ', text)
        text = text.rstrip(' _\n\t')
        text = re.sub(r'\n', ' ', text)
        text = text.lower()
        return text

    def file_loader(self):
        """
        Each file that is loaded is returned as 
        a string of all the xml data inside it.
        """
        for fol in self.config['ft_data_folders']:
            folder = os.path.join(fol, self.split)
            for f in os.listdir(folder):
                filename = os.path.join(folder, f)
                if os.path.isfile(filename) and filename.endswith('.xml'):
                    with open(filename, 'r') as g:
                        yield g.read()

    def refine_xml(self, xml_string):
        xml_string = re.sub(r'<claim [^>]*>', r'<claim>', xml_string)
        xml_string = re.sub(r'<premise [^>]*>', r'<premise>', xml_string)
        return xml_string

    def divide_pc(self, post):
        """
        Divides a post/comment into various contiguous parts 
        corresponding to claims/premises
        """
        type_lis = []
        str_lis = []

        for elem in post:

            s = self.clean_text( str(elem) )
            if s=='':
                continue
            if s.startswith('<claim>'):
                str_lis.append( s[7:-8] )
                type_lis.append(1)
            elif s.startswith('<premise>'):
                str_lis.append( s[9:-10]  )
                type_lis.append(2)
            else:
                str_lis.append( s )
                type_lis.append(0)

        return str_lis, type_lis        

    def thread_generator(self):
        for xml_string in self.file_loader():
            thread, authors, i = [], {}, 0
            xml_string = self.refine_xml(xml_string)
            parsed_xml = BeautifulSoup(xml_string, "xml")
            
            for elem in parsed_xml.find_all('reply')+[parsed_xml.find('OP')]:
                thread.append( self.divide_pc(elem.contents) )
                if len(thread)==self.config['max_tree_size']:
                    break
                
                if 'author' not in elem:
                    thread[-1][0][0] = '<unu> '+thread[-1][0][0]
                    continue

                if elem['author'] not in authors:
                    authors[elem['author']]=i
                    i+=1
                
                author_idx = authors[elem['author']]
                
                thread[-1][0][0] = f'<user_{author_idx}> '+thread[-1][0][0]
                
            yield thread