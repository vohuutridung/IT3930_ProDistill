"""
NOTE: WE DONT USE THIS CUSTOMIZED TRAINER TO DO ANYTHING SUCH AS 
COMPUTE LOSS. THIS IS USED JUST TO GET DATALOADER.
IN THE ORIGINAL REPO, THIS IS JUST A WAY AUTHORS USE TO LEVERAGE EXISTENT CODE
FOR OTHER MODULES.
"""

from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F

class CustomizedTrainer(Trainer):
    def __init__(self, use_multitask_setting: bool=False, *args, **kwargs):
        """
        Customized trainer with user-defined train loss function
        """
        super(CustomizedTrainer, self).__init__(*args, **kwargs)
        self.use_multitask_setting = use_multitask_setting

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False):
        assert 'labels' in inputs, 'labels are not involved in inputs'
        labels = inputs.pop('labels')

        if self.use_multitask_setting:
            assert 'dataset_ids' in inputs.keys(), 'key dataset_ids is missing in inputs!'

            dataset_ids = inputs['dataset_ids']
            outputs = model(**inputs)
            logits = outputs['logits']
            total_loss = None

            for dataset_id in dataset_ids.unique():
                single_dataset_indices = dataset_ids == dataset_id
                single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
                # cross-entropy loss for classification
                if single_dataset_num_labels > 1:
                    loss = F.cross_entropy(input=logits[single_dataset_indices][:, :single_dataset_num_labels], target=labels[single_dataset_indices].long())
                # mse loss for regression
                else:
                    assert single_dataset_num_labels == 1, "wrong number of labels!"
                    loss = F.mse_loss(input=logits[single_dataset_indices][:, 0], target=labels[single_dataset_indices])
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            return (total_loss, outputs) if return_outputs else total_loss
        else:
            outputs = model(**inputs)
            logits = outputs["logits"]
            if logits.shape[1] > 1:
                # cross-entropy loss for classification
                loss = F.cross_entropy(input=logits, target=labels)
            else:
                # mse loss for regression
                assert logits.shape[1] == 1, "wrong number of labels!"
                loss = F.mse_loss(input=logits.squeeze(dim=1), target=labels)
            return (loss, outputs) if return_outputs else loss


glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2
}

glue_data_id_map = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "stsb": 3,
    "qqp": 4,
    "mnli": 5,
    "qnli": 6,
    "rte": 7
}

rev_glue_data_id_map = {value: key for key, value in glue_data_id_map.items()}