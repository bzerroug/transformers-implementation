class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after the last time the validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path to save the model checkpoint when the validation loss improves.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): Validation loss to monitor.
            model (nn.Module): PyTorch model to save when the validation loss improves.
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        if self.verbose:
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')