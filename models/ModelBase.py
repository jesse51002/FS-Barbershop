class Model:
    def __init__(self) -> None:
        self.name = None
        raise NotImplementedError("Init method not implemented and save not set")
    
    def inference(self):
        raise NotImplementedError("Inference method not implemented")