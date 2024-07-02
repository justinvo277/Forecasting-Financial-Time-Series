import torch.nn as nn 
import torch
import utils
import torch.utils
from tqdm import tqdm


def run_encoder_decoder_inference(
    model: nn.Module,
    datatrain: torch.utils.data.Dataset,
    criterion_list: dict, 
    forecast_window: int,
    device: torch.cuda,
    ) -> torch.Tensor:

    loop = tqdm(datatrain, leave=True)
    total_loss_dict = {}

    for name, criterion in criterion_list:
        total_loss_dict[name] = 0.0
    
    model.eval()
    with torch.no_grad():

        for batch_idx, (src, _, trg_y) in enumerate(loop):
            target_seq_dim = 1

            src = src.float()
            src = src.to(device)
            trg = src[:, -1, 0] # shape [batch_size, 1, 1]
            trg = trg.to(device)
            trg = trg.unsqueeze(-1)
            trg = trg.unsqueeze(-1)
            trg_y = trg_y.float()
            trg_y = trg_y.to(device)

            for _ in range(forecast_window-1):
                dim_a = trg.shape[1]
                dim_b = src.shape[1]

                tgt_mask = utils.generate_square_subsequent_mask(
                    dim1=dim_a,
                    dim2=dim_a,
                    )
                tgt_mask = tgt_mask.to(device)

                src_mask = utils.generate_square_subsequent_mask(
                    dim1=dim_a,
                    dim2=dim_b,
                    )
                src_mask = src_mask.to(device)

                # Make prediction
                prediction = model(src, trg, src_mask, tgt_mask) 
                # print(f"Prediction_shape: {prediction.shape}")

                last_predicted_value = prediction[:, -1, :]

                # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
                last_predicted_value = last_predicted_value.unsqueeze(-1)
                # print(f"Last_Predicted: {last_predicted_value.shape}")

                # Detach the predicted element from the graph and concatenate with tgt in dimension 1 or 0
                trg = torch.cat((trg, last_predicted_value.detach()), target_seq_dim)
                trg = trg.to(device)

            # print(f"Final_trg_shape: {trg.shape}")
            # Create masks
            dim_a = trg.shape[1]
            dim_b = src.shape[1] 

            tgt_mask = utils.generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_a,
                )
            tgt_mask = tgt_mask.to(device)

            src_mask = utils.generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_b,
                )
            src_mask = src_mask.to(device)

            # Make final prediction
            final_prediction = model(src, trg, src_mask, tgt_mask)

            for name, criterion in criterion_list:
                loss = criterion(final_prediction, trg_y)
                total_loss_dict[name] += loss.item()
        for name_loss in total_loss_dict:
            total_loss_dict[name_loss] /= len(datatrain)

        return total_loss_dict