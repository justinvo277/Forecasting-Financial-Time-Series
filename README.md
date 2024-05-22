# -DSP391m-Forecasting-Financial-Time-Series-With-Transformer


### Code rules in folders

- Write all methods as functions or classes.
- Describes the input and output data types for each method. For example:
'''
def example_function(images:torch.tensor, alpha:float=1.0) -> torch.tensor:
        """
        Before CutMix/MixUp: images.shape = torch.Size([4, 3, 224, 224]), labels.shape = torch.Size([4])
        After CutMix/MixUp: images.shape = torch.Size([4, 3, 224, 224]), labels.shape = torch.Size([4, 100])
        """
        #### Todo something ###
        return tensor
'''
- Commit descriptor when pushing code to the repo.