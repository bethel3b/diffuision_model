class Trainer:
    def __init__(self, args):
        self.model = args["model"]
        self.train_loader = args["train_loader"]
        self.loss = args["loss"]
        self.optimizer = args["optimizer"]
        self.device = args["device"]
    
    def run(self):
        return