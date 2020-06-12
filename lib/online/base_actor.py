from online import TensorDict
import torch.nn as nn


class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)


        # fix backbone again
        # fix the first three blocks
        print('======> fix backbone again <=======')
        for param in self.net.feature_extractor.parameters():
            param.requires_grad = False
        for m in self.net.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for layer in ['layeronline']:
            for param in getattr(self.net.feature_extractor.features.features, layer).parameters():
                param.requires_grad = True
            for m in getattr(self.net.feature_extractor.features.features, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        print('double check trainable')
        self.check_trainable(self.net)



    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    def check_trainable(self, model):
        """
        print trainable params info
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print('trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        assert len(trainable_params) > 0, 'no trainable parameters'
