from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, folder, step_count, epoch, hp):
        super(Visualizer, self).__init__()
        self.epoch = epoch
        self.step_count = step_count
        self.hp = hp
        self.writer = SummaryWriter(folder, flush_secs=10)

    def plot(self, name, variable):
        self.writer.add_scalar(name + str(self.hp), variable, self.step_count)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
