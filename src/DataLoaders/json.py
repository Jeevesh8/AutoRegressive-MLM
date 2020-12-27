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
        
        for comment in post_tree['comments']:
            if 'body' not in comment:
                post_tree['comments'].remove(comment)
        
        for k in list(post_tree.keys()):        
            if k not in ['title', 'selftext', 'id', 'comments']:
                post_tree.pop(k)
            
        for comment in post_tree['comments']:
            
            if 'replies' in comment and comment['replies'] != '':
                comment['replies'] = comment['replies']['data']['children']
            else:
                comment['replies'] = []
            
            for k in list(comment.keys()):
                if k not in ['replies', 'body', 'id', 'parent_id']:
                    comment.pop(k)
                
        post_tree['comments'] = { comment['id'] : comment for comment in post_tree['comments']}
        return post_tree
    
    def cleaning_pipeline(self, tree):
        tree = self.remove_redundant(tree)
        tree['selftext'] = self.clean_text(tree['selftext'])
        tree['title'] = self.clean_text(tree['title'])
        for id, comment in tree['comments'].items():
            comment['body'] = self.clean_text(comment['body'])
            comment['parent_id'] = comment['parent_id'][3:]
        return tree
    
    def tree_generator(self):
        for data in self.file_loader():
            for tree in data:
                yield self.cleaning_pipeline(tree)
    