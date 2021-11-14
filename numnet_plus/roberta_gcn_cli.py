import os
import json
import options
import argparse
from pprint import pprint
from tools.model import DropBertModel
from mspan_roberta_gcn.roberta_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from tag_mspan_robert_gcn.roberta_batch_gen_tmspan import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet
from datetime import datetime
from tools.utils import create_logger, set_environment
from transformers import RobertaTokenizer, AutoModel
import torch


parser = argparse.ArgumentParser("Bert training task.")
options.add_bert_args(parser)
options.add_model_args(parser)
options.add_data_args(parser)
options.add_train_args(parser)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump(vars(args), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps
logger = create_logger("Bert Drop Pretraining", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)

def main():
    cross_validation_length = 5 # 5 subsets
    cross_avg_f1 = [float("-inf")] * cross_validation_length
    cross_avg_em = [float("-inf")] * cross_validation_length
    # import debugpy
    # logger.info('debug')
    # debugpy.listen(5678)
    # debugpy.wait_for_client()

    for i in range(cross_validation_length):
        logger.info("\n----------------------- Cross Validation with dev index: {} -----------------------\n".format(i))
        best_f1 = float("-inf")
        best_em = float("-inf")
        logger.info("Loading data...")

        

        if not args.tag_mspan:
            train_itr = DropBatchGen(args, data_mode="train", tokenizer=tokenizer, cross_index=i)
            dev_itr = DropBatchGen(args, data_mode="dev", tokenizer=tokenizer, cross_index=i)
        else:
            train_itr = TDropBatchGen(args, data_mode="train", tokenizer=tokenizer)
            dev_itr = TDropBatchGen(args, data_mode="dev", tokenizer=tokenizer)
        num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
        logger.info("Num update steps {}!".format(num_train_steps))

        logger.info("Build bert model.")
        bert_model = AutoModel.from_pretrained(args.roberta_model)

        logger.info("Build Drop model.")
        if not args.tag_mspan:
            network = NumericallyAugmentedBertNet(bert_model,
                    hidden_size=bert_model.config.hidden_size,
                    dropout_prob=args.dropout,
                    use_gcn=args.use_gcn,
                    use_hgt=args.use_hgt,
                    gcn_steps=args.gcn_steps)
        else:
            network = TNumericallyAugmentedBertNet(bert_model,
                                                hidden_size=bert_model.config.hidden_size,
                                                dropout_prob=args.dropout,
                                                use_gcn=args.use_gcn,
                                                gcn_steps=args.gcn_steps)


    
        logger.info("Build optimizer etc...")
        model = DropBertModel(args, network, num_train_step=num_train_steps)

        train_start = datetime.now()
        first = True

        for epoch in range(1, args.max_epoch + 1):
            model.avg_reset()
            if not first:
                train_itr.reset()
            first = False
            logger.info('At epoch {}'.format(epoch))
            for step, batch in enumerate(train_itr):
                model.update(batch)
                if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                    logger.info("Updates[{0:6}] train loss[{1:.5f}] train em[{2:.5f}] f1[{3:.5f}] remaining[{4}]".format(
                        model.updates, model.train_loss.avg, model.em_avg.avg, model.f1_avg.avg,
                        str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                    model.avg_reset()

                # if best_f1 < 0 and model.step % 3000 == 0:
                #     save_prefix = os.path.join(args.save_dir, "checkpoint_best")
                #     model.save(save_prefix, epoch)
                #     logger.info("save model")
            total_num, eval_loss, eval_em, eval_f1 = model.evaluate(dev_itr)
            logger.info(
                "Eval {} examples, result in epoch {}, eval loss {}, eval em {} eval f1 {}.".format(total_num, epoch,
                                                                                                    eval_loss, eval_em,
                                                                                                    eval_f1))

            if eval_f1 > best_f1:
                save_prefix = os.path.join(args.save_dir, "checkpoint_best")
                model.save(save_prefix, epoch)
                best_f1 = eval_f1
                best_em = eval_em
                logger.info("Best eval F1 {}, and eval EM {} at epoch {}".format(best_f1, best_em, epoch))

        cross_avg_f1[i] = best_f1
        cross_avg_em[i] = best_em
        logger.info("done training in {} seconds!".format((datetime.now() - train_start).seconds))

    logger.info("\nCross-validation is done, with average F1: {}, and average EM: {}".format(sum(cross_avg_f1) / cross_validation_length, sum(cross_avg_em) / cross_validation_length))

if __name__ == '__main__':
    main()
