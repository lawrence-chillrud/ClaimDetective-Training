import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaConfig

class RobertaForClaimDetection(nn.Module):

    def __init__(self, n_classes, unfreeze):
        super(RobertaForClaimDetection, self).__init__()
        self.num_labels = n_classes
        #self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=True, config=config)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

        if not unfreeze:
            print("Freezing base layers...\n")
        else:
            print("The base layers of the model were UNFROZEN, therefore they can be optimized.\n")

        for param in self.roberta.base_model.parameters():
            param.requires_grad = unfreeze

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #_, roberta_output, hidden_states, _ = self.roberta(
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs[2]
        #roberta_output = outputs[1]
        #print(self.roberta.pooler(hidden_states[11]) == roberta_output)
        roberta_output = torch.div(torch.add(self.roberta.pooler(hidden_states[11]), self.roberta.pooler(hidden_states[12])), 2)
        #print("roberta last layer size: ", roberta_output.size())
        #print("Hidden states len: ", len(hidden_states))
        #exit()
        output = self.drop(roberta_output)
        '''
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        '''
        return self.out(output)

