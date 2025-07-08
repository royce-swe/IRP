class EarlyStopper:
    def __init__(self, patience=1000, tolerance=1e-8):
        self.patience = patience
        self.tolerance = tolerance
        self.best_f = float('inf')
        self.counter = 0

    def __call__(self, x, f, accept):
        if abs(f - self.best_f) < self.tolerance:
            self.counter += 1
        else:
            self.counter = 0
            self.best_f = f

        if self.counter >= self.patience:
            print(f"Stopping early after {self.counter} stagnant iterations. Best f = {self.best_f:.8f}")
            raise StopIteration
