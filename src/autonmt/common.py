class LanguagePair:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __str__(self):
        return f'{self.src}-{self.tgt}'

    def reversed(self):
        return LanguagePair(self.tgt, self.src)

class CorpusInfo:
    def __init__(self, corpus, datadir, path):
        self.name = corpus
        self.path = path
        self.complete_path = f'{datadir}/{path}' if path and path[0] != '/' else path

    def complete_path_for_pair(self, pair : LanguagePair):
        return f'{self.complete_path}.{pair}'

    def filename_prefix_for_pair(self, pair : LanguagePair):
        return self.complete_path_for_pair(pair).split('/')[-1]