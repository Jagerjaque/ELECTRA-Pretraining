import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Model
from data_load import PretrainingDataset, pad
import argparse
import random, os


random.seed(0)
torch.manual_seed(0)

def train(model, iterator, optimizer, generator_criterion, discriminator_criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        tokens, tokens_with_mask, generator_mask, discriminator_mask = batch
        optimizer.zero_grad()
        generator_logits, generator_y, discriminator_logits, discriminator_y = model(tokens, tokens_with_mask, generator_mask, discriminator_mask)

        generator_logits = generator_logits.view(-1, generator_logits.shape[-1])  # (N*T, VOCAB)
        generator_y = generator_y.view(-1)  # (N*T,)
        generator_loss = generator_criterion(generator_logits, generator_y.long())

        discriminator_logits = discriminator_logits.view(-1)  # (N*T)
        discriminator_y = discriminator_y.view(-1)  # (N*T,)
        discriminator_loss = discriminator_criterion(discriminator_logits, discriminator_y)

        loss = generator_loss + discriminator_loss
        loss.backward()
        optimizer.step()

        if i % 200 == 0 and i != 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
    return

# def eval(model, iterator, f, ner_criterion, pip_re_criterion, tep_re_criterion, trp_re_criterion):
#     model.eval()
#
#     Words, Is_heads, Tags, Y, Y_hat, re_Y, re_Yhat, ner_logit_list, re_logit_list = [], [], [], [], [], [], [], [], []
#     pip_re_Y, pip_re_Yhat = [], []
#     tep_re_Y, tep_re_Yhat = [], []
#     trp_re_Y, trp_re_Yhat = [], []
#     ner_loss_list, re_loss_list, joint_loss_list = [], [], []
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             words, x, is_heads, tags, ner_y, re_y_pip, re_y_tep, re_y_trp, seqlen = batch
#
#             ner_logits, ner_y, ner_yhat, pip_re_logits, pip_re_y_tensor, pip_re_yhat, tep_re_logits, tep_re_y_tensor, tep_re_yhat, trp_re_logits, trp_re_y_tensor, trp_re_yhat = model(x, is_heads, ner_y, re_y_pip, re_y_tep, re_y_trp)
#
#             ner_loss, re_loss, joint_loss = calculate_loss(ner_logits, ner_y, pip_re_logits, pip_re_y_tensor,
#                            tep_re_logits, tep_re_y_tensor, trp_re_logits, trp_re_y_tensor,
#                            ner_criterion, pip_re_criterion, tep_re_criterion, trp_re_criterion)
#
#             ner_loss_list.append(ner_loss)
#             re_loss_list.append(re_loss)
#             joint_loss_list.append(joint_loss)
#
#             Words.extend(words)
#             Is_heads.extend(is_heads)
#             Tags.extend(tags)
#
#             Y.extend(ner_y.cpu().numpy().tolist())
#             Y_hat.extend(ner_yhat.cpu().numpy().tolist())
#
#             pip_re_Y.extend(val for val in pip_re_y_tensor.cpu().numpy().tolist())
#             pip_re_Yhat.extend(val for val in pip_re_yhat.cpu().numpy().tolist())
#
#             tep_re_Y.extend(val for val in tep_re_y_tensor.cpu().numpy().tolist())
#             tep_re_Yhat.extend(val for val in tep_re_yhat.cpu().numpy().tolist())
#
#             trp_re_Y.extend(val for val in trp_re_y_tensor.cpu().numpy().tolist())
#             trp_re_Yhat.extend(val for val in trp_re_yhat.cpu().numpy().tolist())
#
#     pip_re_Y = [z for y in pip_re_Y for z in y]
#     pip_re_Yhat = [z for y in pip_re_Yhat for z in y]
#
#     tep_re_Y = [z for y in tep_re_Y for z in y]
#     tep_re_Yhat = [z for y in tep_re_Yhat for z in y]
#
#     trp_re_Y = [z for y in trp_re_Y for z in y]
#     trp_re_Yhat = [z for y in trp_re_Yhat for z in y]
#
#     for pip_yhat, tep_yhat, trp_yhat in zip(pip_re_Yhat, tep_re_Yhat, trp_re_Yhat):
#         if pip_yhat and not tep_yhat and not trp_yhat:
#             re_Yhat.append(re_tag2idx[pip_idx2tag[pip_yhat]])
#         elif not pip_yhat and tep_yhat and not trp_yhat:
#             re_Yhat.append(re_tag2idx[tep_idx2tag[tep_yhat]])
#         elif not pip_yhat and not tep_yhat and trp_yhat:
#             re_Yhat.append(re_tag2idx[trp_idx2tag[trp_yhat]])
#         else:
#             re_Yhat.append(re_tag2idx['O'])
#
#     for pip_y, tep_y, trp_y in zip(pip_re_Y, tep_re_Y, trp_re_Y):
#         if pip_y and not tep_y and not trp_y:
#             re_Y.append(re_tag2idx[pip_idx2tag[pip_y]])
#         elif not pip_y and tep_y and not trp_y:
#             re_Y.append(re_tag2idx[tep_idx2tag[tep_y]])
#         elif not pip_y and not tep_y and trp_y:
#             re_Y.append(re_tag2idx[trp_idx2tag[trp_y]])
#         else:
#             re_Y.append(re_tag2idx['<PAD>'])
#
#     re_Yhat = [yhat for y, yhat in zip(re_Y, re_Yhat) if y != 0]
#     re_Y = [y for y in re_Y if y != 0]
#
#     conf_mx = confusion_matrix(y_true=re_Y, y_pred=re_Yhat, labels=range(1, len(RE_VOCAB)))
#     conf_mx_file = open(f+'_confusion_matrix', 'w')
#     conf_mx_file.write(str(RE_VOCAB[1:])+'\n')
#     conf_mx_file.write(str(conf_mx))
#     re_metrics = classification_report(re_Y, re_Yhat, labels=range(2, len(RE_VOCAB)), target_names=RE_VOCAB[2:], digits=5)
#     print(re_metrics)
#     open(f+'_re_results.txt', 'w').write(re_metrics)
#
#     re_micro_f1 = classification_report(re_Y, re_Yhat, labels=range(2, len(RE_VOCAB)), target_names=RE_VOCAB[2:], output_dict=True)['micro avg']['f1-score']
#
#     letters = string.ascii_lowercase
#     temp_file_name = ''.join(random.choice(letters) for i in range(10))
#
#     # gets results and save
#     with open(temp_file_name, 'w') as fout:
#         for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
#             y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
#             preds = [ner_idx2tag[hat] for hat in y_hat]
#             if len(preds) == len(words.split()) == len(tags.split()):
#                 for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
#                     if t is not '<PAD>':
#                         fout.write(f"{w} {t} {p}\n")
#             else:
#                 fout.write('\n')
#             fout.write("\n")
#
#     with open(temp_file_name) as fout:
#         conll_metrics, performance_string = evaluate_conll_file(fout)
#
#     final = f + ".P%.2f_R%.2f_F%.2f" % (conll_metrics[0], conll_metrics[1], conll_metrics[2])
#     with open(final, 'w') as fout:
#         fout.write(performance_string)
#
#     os.remove(temp_file_name)
#
#     return conll_metrics[0], conll_metrics[1], conll_metrics[2], re_micro_f1, re_metrics, ner_loss_list, re_loss_list, joint_loss_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--training_directory', type=str, default='./data/')
    hp = parser.parse_args()

    device = hp.device
    train_dataset = PretrainingDataset(hp.training_directory)
    model = Model(device)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, betas=(hp.beta1, hp.beta2), eps=hp.epsilon, weight_decay=hp.weight_decay)
    generator_criterion = nn.CrossEntropyLoss(ignore_index=0) # don't backprop error from the [PAD] token
    discriminator_criterion = nn.BCEWithLogitsLoss()

    print('starting training')
    for epoch in range(1, hp.n_epochs + 1):
        print(f"========= training on epoch {epoch}=========")
        train(model, train_iter, optimizer, generator_criterion, discriminator_criterion, device)

    print(f"========= saving ELECTRA Discriminator TAPT weights at {hp.logdir}=========")
    torch.save(model.discriminator.electra.state_dict(), os.path.join(f'{hp.logdir}', 'pretrained_weights.bin'))
