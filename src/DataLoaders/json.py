import jsonlist
import re

class load_reddit_data:
    
    def __init__(self, config):
        self.config = config
        
    def file_loader(self):
        for f in self.config['data_files']:
            yield jsonlist.load_file(f)
    
    def clean_text(self, text):
        
        text = text.strip(' _\t\n')
        text = text.split('____')[0]                                                    #To remove footnotes
        text = text.strip(' _\t\n')
        text = re.sub(r'\(https?://\S+\)', '<url>', text)                               #To remove URLs
        text = re.sub(r'\n', '', text)
        text = re.sub(r'&gt;.*(?!(\n+))$', '', text)                                    #To remove quotes at last.
        text = text.rstrip(' _\n\t')
        text = re.sub(r'\n', '', text)
        return text
    
    def remove_redundant(self, post_tree):
        """
        Removes redundant keys from the data dictionary.
        And changes the the comments list into a dictionary with the comment_id as key.
        """

        for k in list(post_tree.keys()):        
            if k not in ['title', 'selftext', 'id', 'comments']:
                del post_tree[k]
            
        for comment in post_tree['comments']:
                
            if comment['replies'] != '':
                comment['replies'] = comment['replies']['data']['children']
            else:
                comment['replies'] = []
            
            for k in list(comment.keys()):
                if k not in ['replies', 'body', 'id', 'parent_id']:
                    del comment[k]
                
        post_tree['comments'] = { comment['id'] : comment for comment in post_tree['comments']}

        return post_tree
    
    def cleaning_pipeline(self, tree):
        tree = self.remove_redundant(tree)
        tree['selftext'] = self.clean_text(tree['selftext'])
        tree['title'] = self.clean_text(tree['title'])
        for id, comment in tree['comments']:
            comment['body'] = self.clean_text(comment['body'])
            comment['parent_id'] = comment['parent_id'][3:]
        return tree
    
    def tree_generator(self):
        for data in self.file_loader():
            for tree in data:
                yield self.cleaning_pipeline(tree)
    

    '''
    def make_tree(self, post_tree):
        for id, comment in post_tree['comments']:
            comment['children'] = {}
            i=0
            for reply_id in comment['replies']: 
                comment['children'][f'child_{i}'] = post_tree['comments'][reply_id]
            
    def make_text_tree(self, post_tree):
        text_tree = {}
        i=0
        text_tree['title'] = post_tree['title']
        text_tree['post_content'] = post_tree['selftext']
        text_tree['child_count'] = 0
        text_tree['children'] = {}
        for id, comment in post_tree['comments'].items():
            if comment['parent_id'][3:]==post_tree[id]:
                child_name = f'child_{text_tree['child_count']}'
                text_tree['child_count'] += 1
                text_tree['children'][child_name] = {'text' : comment['body'], 'child_count' : 0, 'children'={}, 'id' : comment['id']}

        if type(post_tree) is str :

        
        text_tree = {}
        if top_level:
            
            for i in range(len(post_tree['comments'])):
                
                text_tree[f'comment_{i}_text'] = post_tree['comments'][i]['body']

                text_tree[f'comment_{i}_subcomments'] = self.make_text_tree(post_tree['comments'][i]['replies'],
                                                                            top_level=False)
            return text_tree
        
        else :



    def get_post_tree(self):
        for tree in self.data:
            tree = self.make_text_tree(tree)

    def get_encoder_batch(self) :
        for tree in self.data :
    
    #def json_to_text_tree()
