import torch

from probts.utils.constant import PROBTS_DATA_KEYS


class ProbTSBatchData:
    input_names_ = PROBTS_DATA_KEYS

    def __init__(self, data_dict, device):
        self.__dict__.update(data_dict)

        # TODO: merge with fill_inputs input_names_
        if 'context_length' in data_dict:
            self.__dict__['context_length'] = data_dict['context_length']
        if 'prediction_length' in data_dict:
            self.__dict__['prediction_length'] = data_dict['prediction_length']
            
        if 'max_context_length' in data_dict:
            self.__dict__['max_context_length'] = data_dict['max_context_length']
        else:
            self.__dict__['max_context_length'] = None
            
        if 'max_prediction_length' in data_dict:
            self.__dict__['max_prediction_length'] = data_dict['max_prediction_length']
        else:
            self.__dict__['max_prediction_length'] = None

        if len(self.__dict__["past_target_cdf"].shape) == 2:
            self.expand_dim()
        self.set_device(device)
        self.fill_inputs()
        self.process_pad()

    def fill_inputs(self):
        for input in self.input_names_:
            if input not in self.__dict__:
                self.__dict__[input] = None

    def set_device(self, device):
        for k, v in self.__dict__.items():
            if v is not None and torch.is_tensor(v):
                v.to(device)
        self.device = device

    def expand_dim(self):
        self.__dict__["target_dimension_indicator"] = self.__dict__[
            "target_dimension_indicator"
        ][:, :1]
        for input in [
            "past_target_cdf",
            "past_observed_values",
            "future_target_cdf",
            "future_observed_values",
        ]:
            self.__dict__[input] = self.__dict__[input].unsqueeze(-1)

    def process_pad(self):
        if self.__dict__["past_is_pad"] is not None:
            self.__dict__["past_observed_values"] = torch.min(
                self.__dict__["past_observed_values"],
                1 - self.__dict__["past_is_pad"].unsqueeze(-1),
            )
