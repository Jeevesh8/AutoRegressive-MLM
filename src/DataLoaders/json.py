import jsonlist
import re, os

class load_reddit_data:
    
    def __init__(self, config, mode='train'):
        self.config = config
        self.mask_dms = 'discourse_markers_file' in config
        self.data_file = 'train_period_data.jsonlist' if mode=='train' else 'heldout_period_data.jsonlist'
        if self.mask_dms:
            self.dms = self.get_discourse_markers(config['discourse_markers_file'])

    def file_loader(self):
        for folder in self.config['data_folders']:
            f = os.path.join(folder,self.data_file)
            yield jsonlist.load_file(f)
    
    def mask_disc_markers(self, text):
        punctuations = ".?!;:-()\'\"[]"
        for elem in punctuations:
            text = text.replace(elem, ' '+elem+' ')
        text = ' '+text+' '
        for dm in self.dms:
            text.replace(' '+dm+' ', ' <mask> '*len(dm.split()))
        return text

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
        if self.mask_dms:
            text = self.mask_disc_markers(text)
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
            if k not in ['title', 'selftext', 'id', 'comments', 'author']:
                post_tree.pop(k)
            
        for comment in post_tree['comments']:
            
            if 'replies' in comment and comment['replies'] != '' and comment['replies'] is not None:
                comment['replies'] = comment['replies']['data']['children']
            else:
                comment['replies'] = []
            
            for k in list(comment.keys()):
                if k not in ['replies', 'body', 'id', 'parent_id', 'author']:
                    comment.pop(k)
                
        post_tree['comments'] = { comment['id'] : comment for comment in post_tree['comments']}
        return post_tree
    
    def cleaning_pipeline(self, tree):
        tree = self.remove_redundant(tree)
        tree['selftext'] = self.clean_text(tree['selftext'])
        tree['title'] = self.clean_text(tree['title'])
        
        empty_comments = []
        for id, comment in tree['comments'].items():
            if 'body' in comment and 'parent_id' in comment:
                comment['body'] = self.clean_text(comment['body'])
                comment['parent_id'] = comment['parent_id'][3:]
            else: 
                empty_comments.append(id)
                print('Skipping empty comment : ', id, tree['comments'][id])

        empty_comments_dict = {}
        for id in empty_comments:
            empty_comments_dict[id] = tree['comments'][id]
            tree['comments'].pop(id)
        
        #Children of empty comments are assigned to their parents
        for id, comment in tree['comments'].items():
            parent_id = comment['parent_id']
            while parent_id in empty_comments:
                parent_id = empty_comments_dict[parent_id]['parent_id'][3:]
            comment['parent_id'] = parent_id if parent_id in tree['comments'] else tree['id']
                
        return tree
    
    def new_branch_tree(self, tree, ids):
        """
        Returns a new tree, consisting of first few comments, 
        specified by ids and the OP's text & title.
        """
        branch_tree = {}
        branch_tree['selftext'] = tree['selftext']
        branch_tree['title'] = tree['title']
        branch_tree['id'] = tree['id']
        branch_tree['comments'] = {}
        for id in ids[1:]:
            branch_tree['comments'][id] = tree['comments'][id]
        return branch_tree
    
    def add_children(self, tree, branch_tree, id):
        """
        Adds entire comments subtree rooted at id in tree, to branch_tree.
        """
        if id in tree['comments']:
            branch_tree['comments'][id] = tree['comments'][id]
            for id in branch_tree['comments'][id]['replies']:
                self.add_children(tree, branch_tree, id) 
        
    def branch_generator(self, tree, ids, init_tree):
        """
        Picks a branch followed by subtree such that total size
        is less than config['max_tree_size']
        """
        if len(tree['comments'])<=self.config['max_tree_size']:
            yield tree
        else :
            top_level_comments = []
            for id, comment in tree['comments'].items():
                if comment['parent_id']==ids[-1]:
                    top_level_comments.append(id)
            
            if len(top_level_comments)==0:
                yield None

            for id in top_level_comments:
                branch_tree = self.new_branch_tree(init_tree, ids)
                self.add_children(tree, branch_tree, id)
                ids.append(id)
                for sub_branch in self.branch_generator(branch_tree, ids, init_tree):
                    yield sub_branch
                ids = ids[:-1]
            
    def tree_generator(self):
        for data in self.file_loader():
            for tree in data:
                tree = self.cleaning_pipeline(tree)
                for branch in self.branch_generator(tree, [tree['id']], tree):
                    if branch is not None:
                        yield branch
    
    def get_discourse_markers(self, filename):
        with open(filename) as f:
            dms = f.readlines()[2:]
            dms = [elem.split(' ', 1)[1].rstrip('\n') for elem in dms]
        return dms

    def get_sentences(self):
        """
        Yields all sentences in all posts/comments one-by-one.
        """
        for tree in self.tree_generator():
            yield tree['title'] + ' ' + tree['selftext']
            for id, comment in tree['comments'].items():
                yield comment['body']
