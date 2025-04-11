import torch
import editdistance

from src.data.text.common_token_txt import CTC_PAD


def groupe_features_per_class(features, gt_seq_frames, index_class_to_filter):
    dict_feature_per_class = {}

    for features_one_item, y_one_item in zip(features, gt_seq_frames):

        if y_one_item is None:
            continue
        # y_one_item: tensor
        for f, y in zip(features_one_item, y_one_item):
            if y.item() in index_class_to_filter:
                continue
            else:
                if y.item() == CTC_PAD:
                    print("Error Pad class is used")
                else:
                    if y.item() in dict_feature_per_class:
                        dict_feature_per_class[y.item()].append(f)
                    else:
                        dict_feature_per_class[y.item()] = [f]

    return dict_feature_per_class


def compute_center_loss_k1(dict_features_per_class, clusters, loss_fct):
    loss_center_all_class = 0

    nb_class = 0
    for index_class, features in dict_features_per_class.items():
        nb_frames_used_class = 0
        loss_one_class = 0
        for one_feature in features:
            index_class_loss = index_class

            loss_reg = loss_fct(one_feature, clusters[index_class_loss])

            loss_one_class += loss_reg
            nb_frames_used_class += 1

        # Norm per class, not all item because classes are unbalanced
        if nb_frames_used_class != 0:
            loss_one_class /= nb_frames_used_class
            nb_class += 1

        loss_center_all_class += loss_one_class

    if nb_class != 0:
        loss_center_all_class /= nb_class

    return loss_center_all_class


def compute_center_coordinates(data_loader,
                               model,
                               device,
                               char_list,
                               token_blank,
                               index_class_to_filter,
                               text_read):
    model.eval()

    prototypes_after = torch.zeros([len(char_list), 512]).to(device)

    dict_feature_per_class_after = {}

    # Get prediction features
    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

            y_gt_txt = batch_data["label_str"]

            nb_item_batch = x.shape[0]

            y_pred, _, after_blstm = model(x)

            after_blstm = torch.permute(after_blstm, (1, 0, 2))
            after_blstm = torch.sigmoid(after_blstm)

            # Encoder
            encoder_outputs_main, encoder_outputs_shortcut = y_pred
            encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            encoder_outputs_main = encoder_outputs_main.transpose(0, 1)

            top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(encoder_outputs_main)]
            predictions_text_main_enc = [text_read.ctc_best_path_one(p, char_list, token_blank) if p is not None else ""
                                         for p in
                                         top_main_enc]

            cers_enc = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

            for i in range(nb_item_batch):
                if cers_enc[i] == 0:

                    # Group features by character
                    for f, y in zip(after_blstm[i], top_main_enc[i]):
                        if y in index_class_to_filter:
                            continue
                        if y in dict_feature_per_class_after:
                            dict_feature_per_class_after[y].append(f)
                        else:
                            dict_feature_per_class_after[y] = [f]

    # Compute means
    for key in dict_feature_per_class_after:
        if len(dict_feature_per_class_after[key]) > 0:
            # N, nb features
            features_tensor = torch.stack(dict_feature_per_class_after[key])

            mean_value = torch.mean(features_tensor, 0)
            mean_value = mean_value.detach()

            prototypes_after[key] = mean_value

    return prototypes_after
