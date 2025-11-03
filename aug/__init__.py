'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
from .methods import *
from .preprocess_cm import *
import logging


class Method:
    def __init__(self, args):
        self.method = args.method
        self.args = args
        self.path = self.args.save_path
        self.logger = config_logger(self.path + '.log')
        self.logger.info(args)
        self.logger.info("Output path: " + self.path)
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.args.feature == 'origin_darpa':
            # dataset = get_dataset_darpatc3(self.args.dataset_darpa)
            dataset = get_dataset_darpatc3_ego1(self.args.dataset_darpa, h=self.args.ego)
        else:
            raise NotImplementedError
        self.logger.info("Dataset is ok.")
        return dataset

    def train(self):
        # GraphCL
        logging.info('Start training')
        if self.method == 'GraphCL':
            return GraphCL(self.args.times, self.args, self.path, self.logger, self.dataset, False, False)
        elif self.method == 'GraphCL_OGSN':
            return GraphCL(self.args.times, self.args, self.path, self.logger, self.dataset, True, False)
        elif self.method == 'GraphCL_OGSN_ATT':
            return GraphCL(self.args.times, self.args, self.path, self.logger, self.dataset, False, True)
        # JOAO
        elif self.method == 'JOAO':
            return JOAO(self.args.times, self.args, self.path, self.logger, self.dataset, False, False)
        elif self.method == 'JOAO_OGSN':
            return JOAO(self.args.times, self.args, self.path, self.logger, self.dataset, True, False)
        elif self.method == 'JOAO_OGSN_ATT':
            return JOAO(self.args.times, self.args, self.path, self.logger, self.dataset, False, True)
        else:
            raise NotImplementedError


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger

