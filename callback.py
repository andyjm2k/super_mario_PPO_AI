import environment_pyboy_neat_mario as env


class TrainAndLoggingCallback(env.BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path:
            env.os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = env.os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True
