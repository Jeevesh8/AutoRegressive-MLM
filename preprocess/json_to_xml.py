import argparse
import json, os
import jsonlist
from functools import reduce

def clean_claim(claim_text_lis):
    """
    Returns a list of claims with metadata removed.
    Don't call multiple times, on same lis.
    """
    clean_claims = []
    for claim_text in claim_text_lis:
        if claim_text is not None:
            clean_claim = ' '.join(claim_text.split()[1:]).strip(' ')                      #Remove initial number
            if clean_claim!='':
                clean_claims.append( clean_claim )                               
    
    return clean_claims

def clean_premise(premise_lis):
    """
    Returns a list of premises with meta data removed.
    """
    if type(premise_lis) is dict:
        premise_lis = reduce(lambda x, y: x+y, [v if v is not None else [] for k,v in premise_lis.items()], [])
    clean_premises = []
    for lis in premise_lis:
        if lis is not None:
            clean_premises += clean_claim(lis)
    return clean_premises

def mark_comment(comment, claim_lis=None, premise_lis=None):
    """
    Adds <claim>/<premise> tags to comment.
    """
    comment = ' '.join(comment.split(' '))

    comment = ' '+comment+' '
    
    if claim_lis is not None:
        for claim in claim_lis:
            claim = ' '.join(claim.split(' '))
            claim = claim.strip(' ')
            print("Replacing CLAIM : ", claim)
            comment = comment.replace(claim, '<claim>'+claim+'</claim>')
    
    if premise_lis is not None:
        for premise in premise_lis:
            premise = ' '.join(premise.split(' '))
            premise = premise.strip(' ')
            print("Replacing PREMISE : ", premise)
            comment = comment.replace(premise, '<premise>'+premise+'</premise>')
    
    return comment[1:-1]

def format_annotation(annotation, post_tree):
    """
    Modifies annotation to add claim and premise tags and returns xml.
    """
    xml_out = ''
    comment_ids = [elem['id'] for elem in post_tree['comments']]
    
    comment1_id = annotation['Comment1']
    comment2_id = annotation['Comment2']
    
    #Preparing XML for Comment 1
    if comment1_id in comment_ids:
        cur_comment = post_tree['comments'][comment_ids.index(comment1_id)]
        if 'ann_claim_premise' not in cur_comment:
            cur_comment['ann_claim_premise'] = mark_comment(cur_comment['body'],
                                                            clean_claim(annotation['Claim1']) if 'Claim1' in annotation else None,
                                                            clean_premise(annotation['Premise1']) if 'Premise1' in annotation else None)
            
        xml_out += '<reply>'+cur_comment['ann_claim_premise']+'</reply>'
    
    elif comment1_id == post_tree['id']:
        if 'ann_claim_premise' not in post_tree:
            post_tree['ann_claim_premise'] = mark_comment(post_tree['selftext'],
                                                          clean_claim(annotation['Claim1']) if 'Claim1' in annotation else None,
                                                          clean_premise(annotation['Premise1']) if 'Premise1' in annotation else None)

        xml_out += '<OP>'+post_tree['ann_claim_premise']+'</OP>'
    
    else:
        raise AssertionError("Comment id : ", comment1_id, " not found in the post tree : ", post_tree)
    
    #Preparing XML for Comment 2
    if comment2_id in comment_ids:
        cur_comment = post_tree['comments'][comment_ids.index(comment2_id)]
        
        if 'ann_claim_premise' not in cur_comment:
            cur_comment['ann_claim_premise'] = mark_comment(cur_comment['body'],
                                                            clean_claim(annotation['Claim2']) if 'Claim2' in annotation else None,
                                                            clean_premise(annotation['Premise2']) if 'Premise2' in annotation else None)
        
        xml_out += '<reply>'+cur_comment['ann_claim_premise']+'</reply>'
    else:
        raise AssertionError("Comment id : ", comment2_id, " not found in the post tree : ", post_tree)
    
    return xml_out

def get_next_file_name(write_dir):
    i = 0
    file_name = os.path.join(write_dir, str(i)+'.xml')
    while True:
        while os.path.isfile(file_name):
            i+=1
            file_name = os.path.join(write_dir, str(i)+'.xml')
        yield file_name

def write_xml(thread_xml, write_dir, file_name_iter):
    xml_content = """<?xml version="1.0"?> <thread> \n"""
    for elem in thread_xml:
        xml_content+=(elem+'\n')
    xml_content+="</thread>"

    with open(next(file_name_iter), 'w')  as f:
        f.write(xml_content)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_file', type=str, help='Json file with annotations to be converted to XML file')
    parser.add_argument('--reddit_file', type==str, help='Jsonlist File with reddit comments; in the format of data of https://chenhaot.com/pages/changemyview.html.\
                                                        This file will be searched for comments matching those in json_file')
    parser.add_argument('--write_dir', type=str, help='Directory to which the program should write the generated xml files.')
    
    args = parser.parse_args()
    
    file_name_iter = get_next_file_name(args.write_dir)
    
    with open(args.json_file, 'r') as f:
        ft_data = json.load(f)

    print("Loaded finetuning data")
    train_data = jsonlist.load_file(jsonlist.load_file(args.reddit_file))

    annotations = []
    for key in ft_data.keys():
        for k, annotation in ft_data[key].items():
            annotations.append(annotation)
    annotations = annotations[:-10]                                                     #Remove Last 10 annotations, they have discrepancy b/w Claim1 and Claim2

    post_ids = [elem['id'] for elem in train_data]

    post_comment_ids = [ elem['id'] for elem in train_data ]
    parent_post_ids = { elem : elem for elem in post_comment_ids }
    for elem in train_data:
        for c in elem['comments']:
            post_comment_ids += [ c['id'] ]
            parent_post_ids[c['id']] = elem['id']


    i = 0
    while i<len(annotations) :

        annotation = annotations[i]
        post_comment_id = annotation['Comment1']
        thread = []
        thread_xml = []
        
        try :
            parent_post = parent_post_ids[post_comment_id]
            idx = post_ids.index(parent_post)

        except KeyError:
            raise KeyError("Can't find post/comment of id : ", post_id)
                
        comment_ids = [elem['id'] for elem in train_data[idx]['comments']]
            
        while True:
            
            comment_id = annotation['Comment2']
                
            if comment_id in comment_ids:
                thread_xml.append( format_annotation(annotation, train_data[idx]) )
                thread.append( annotation )
            else :
                raise ValueError("Invalid comment id: ", comment_id, " for post with id : ", parent_post)
                
            i+=1
            
            if i==len(annotations):
                break
            
            annotation = annotations[i]

            if annotation['Comment1']!=comment_id or i==len(annotations):
                write_xml(thread_xml)
                
                if annotation['Comment1'] not in [a['Comment2'] for a in thread]:
                    break               
                
                while True:
                    if thread[-1]['Comment2']!=annotation['Comment1']:
                        thread.pop()
                        thread_xml.pop()
                    else:
                        break
