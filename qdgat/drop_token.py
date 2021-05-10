from allennlp.data.tokenizers import Token as OToken


class Token(OToken):
    def __new__(cls, text: str = None,
        idx: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        edx: int = None
    ):
        self = super(Token, cls).__new__(cls)
        self.text = text
        self.idx = idx
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id =  text_id
        self.edx = edx
        return self
