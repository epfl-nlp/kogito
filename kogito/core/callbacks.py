class Callback:
    pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __call__(self, model: KnowledgeModel):
        pass