from transformers import AutoTokenizer, BertTokenizer

class PreTrainedTokenizer:
    def __init__(self, tokenizer_type, max_length, truncation,  padding):
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.tensors = "pt"
        self.truncation = truncation
        self.padding = padding
        self.tokenizer = None

    def set_tokenizer(self):
        if self.tokenizer_type == "autotokenizer_best_best_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif self.tokenizer_type == "berttokenizer_best_best_uncased":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def tokenize_messages(self, messages):        
        self.set_tokenizer()
        encoded_input = self.tokenizer(messages, 
                                      padding=self.padding, 
                                      return_tensors=self.tensors, 
                                      truncation=self.truncation, 
                                      max_length=self.max_length
                                      )

        return encoded_input.input_ids
