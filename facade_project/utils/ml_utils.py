class MetricHandler:
    """
    Object meant to be used in the training loop to handle metrics logs
    """

    def __init__(self):
        pass

    def add(self, outputs, targets):
        """
        Adding metric for each batch

        :param outputs: outputs of the model
        :param targets: targets of the model
        """
        raise NotImplementedError()

    def compute(self, phase, dataset_size):
        """
        Aggregate accumulated metrics over batches at the end of the epoch

        :param phase: either 'train' or 'val'
        :param dataset_size: size of the dataset
        """
        raise NotImplementedError()

    def description(self, phase):
        """
        Description of the current metrics

        :param phase: either 'train' or 'val'
        :return: str
        """
        raise NotImplementedError()

    def scalar_infos(self, phase):
        """
        Return list of tuple to use with tensorboard writer object 'add_scalar' function

        :param phase: either 'train' or 'val'
        :return: [tuple(str, number)]
        """
        raise NotImplementedError()

    def description_best(self):
        """
        Description of the best metrics

        :return: str
        """
        raise NotImplementedError()


class Epocher:
    """
    An object which is used to print information about training without spamming the console. (WIP)
    """
    def __init__(self, n_epoch, epoch_offset=1):
        # epoch_offset += 1 # starting at 1 and not zero
        self.n_epoch = n_epoch
        self.epoch_offset = epoch_offset
        self.s_more = ''
        self.stats_string = ''
        self.ls_string = ''

    def __iter__(self):
        self.n = self.epoch_offset - 1
        self.stats_string = ''
        self.ls_string = ''
        self.s_more = ''
        self.__update_stdout__()
        return self

    def __next__(self):
        self.n += 1
        if self.n >= self.n_epoch + self.epoch_offset:
            raise StopIteration
        self.__update_stdout__()
        self.s_more = ''
        return self.n

    def update_stats(self, s):
        self.stats_string = s
        self.__update_stdout__()

    def update_last_saved(self, s):
        self.ls_string = s
        self.__update_stdout__()

    def print(self, s, sep=' '):
        self.s_more = sep + s.replace('\n', '')
        self.__update_stdout__()

    def __update_stdout__(self):
        s0 = 'Epoch [{}/{}]'.format(self.n, self.n_epoch + self.epoch_offset - 1)
        s1, s2 = '', ''
        if self.stats_string != '':
            s1 = ' Stats [{}]'.format(self.stats_string).replace('\n', '')
        if self.ls_string != '':
            s2 = ' Last Saved [{}]'.format(self.ls_string).replace('\n', '')
        print('\r{}'.format(s0), s1, s2, self.s_more, end='', sep='')
