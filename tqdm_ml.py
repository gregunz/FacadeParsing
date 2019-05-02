from tqdm import tqdm as tqdm

def tqdm_if(it, enable=True, **kwargs):
    if enable:
        return tqdm(it, **kwargs)
    return it
   
class Epocher:
    def __init__(self, n_epoch, epoch_offset=1):
        #epoch_offset += 1 # starting at 1 and not zero
        self.n_epoch = n_epoch
        self.epoch_offset = epoch_offset
        
    def __iter__(self):
        self.n = self.epoch_offset - 1
        self.stats_string = ''
        self.ls_string = ''
        self.s_more = ''
        self.update_stdout()
        return self

    def __next__(self):
        self.n += 1
        if self.n >= self.n_epoch + self.epoch_offset:
            raise StopIteration
        self.update_stdout()
        self.s_more = ''
        return self.n
        
    def update_stats(self, s):
        self.stats_string = s
        self.update_stdout()
        
    def update_ls(self, s):
        self.ls_string = s
        self.update_stdout()
        
    def update_stdout(self):
        s0 = 'Epoch [{}/{}]'.format(self.n, self.n_epoch + self.epoch_offset - 1)
        s1, s2 = '', ''
        if self.stats_string != '':
            s1 = ' Stats [{}]'.format(self.stats_string).replace('\n', '')
        if self.ls_string != '':
            s2 = ' Last Saved [{}]'.format(self.ls_string).replace('\n', '')
        print('\r{}'.format(s0), s1, s2, self.s_more, end='', sep='')
        
    def print(self, s, sep=' '):
        self.s_more = sep + s.replace('\n', '')
        self.update_stdout()